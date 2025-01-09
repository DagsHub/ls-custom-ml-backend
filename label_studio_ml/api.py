from dagshub.mlflow import get_mlflow_model
from dagshub.data_engine import datasources
from dotenv import load_dotenv

import hmac
import json
import logging
import os
import dagshub
import mlflow
import base64
import tempfile
import cloudpickle
import importlib
import sys

import subprocess
from flask import Flask, request, jsonify, Response

from .response import ModelResponse
from .model import LabelStudioMLBase
from .exceptions import exception_handler

logger = logging.getLogger(__name__)

_server = Flask(__name__)
BASIC_AUTH = None

tempdir = tempfile.mkdtemp()

def init_app(model_instance, basic_auth_user=None, basic_auth_pass=None):
    global model
    global BASIC_AUTH

    model = model_instance
    basic_auth_user = basic_auth_user or os.environ.get('BASIC_AUTH_USER')
    basic_auth_pass = basic_auth_pass or os.environ.get('BASIC_AUTH_PASS')
    if basic_auth_user and basic_auth_pass:
        BASIC_AUTH = (basic_auth_user, basic_auth_pass)

    return _server

@_server.post('/configure')
@exception_handler
def _configure():
    def reverse_lookup(dictionary, target):
        for key in dictionary:
            if target.lower() in [x.lower() for x in dictionary[key]]:
                return key

    global tempdir

    load_dotenv()
    args = json.loads(request.get_json())

    dagshub.auth.add_app_token(args['authtoken'])
    dagshub.init(*args['repo'].split('/')[::-1])  # user-level privileged auth token

    model_uri = f'models:/{args["model"]}/{args["version"]}'
    req_path = mlflow.artifacts.download_artifacts(f'{model_uri}/requirements.txt')
    try:
        uv_output = subprocess.run(f'yes | uv pip install --upgrade -r {req_path}', shell=True, capture_output=True)
    except:
        raise RuntimeError("Failed to install requirements.txt.")
    importlib.invalidate_caches()

    lookup_table = importlib.metadata.packages_distributions()
    for module in [x[3:x.index(b'=')].decode('utf-8') for x in uv_output.stderr.split(b'\n') if b'+' in x]:
        try:
            lib = importlib.import_module(reverse_lookup(lookup_table, module))
            importlib.reload(lib)
        except:
            print(f"Could not re-load {module}")
    importlib.invalidate_caches()

    mlflow_model = get_mlflow_model(args['repo'], args['name'], args['host'], args['version'])
    ds = datasources.get_datasource(args['datasource_repo'], args['datasource_name'])
    dp_map = ds.all().dataframe[['path', 'datapoint_id']]
    model.configure(mlflow_model, *[cloudpickle.loads(base64.b64decode(args[hook])) for hook in ['pre_hook', 'post_hook']], ds, dp_map)
    # model.api = dagshub.common.api.repo.RepoAPI(f'https://dagshub.com/{args["username"]}/{args["repo"]}', host=args['host'])

    return []

@_server.route('/predict', methods=['POST'])
@exception_handler
def _predict():
    """
    Predict tasks

    Example request:
    request = {
            'tasks': tasks,
            'model_version': model_version,
            'project': '{project.id}.{int(project.created_at.timestamp())}',
            'label_config': project.label_config,
            'params': {
                'login': project.task_data_login,
                'password': project.task_data_password,
                'context': context,
            },
        }

    @return:
    Predictions in LS format
    """
    data = request.json
    tasks = data.get('tasks')
    label_config = data.get('label_config')
    project = data.get('project')
    project_id = project.split('.', 1)[0] if project else None
    params = data.get('params', {})
    context = params.pop('context', {})

    model.project_id = project_id
    model.use_label_config(label_config)

    # model.use_label_config(label_config)

    response = model.predict(tasks, context=context, **params)

    # if there is no model version we will take the default
    if isinstance(response, ModelResponse):
        if not response.has_model_version():
            mv = model.model_version
            if mv:
                response.set_version(str(mv))
        else:
            response.update_predictions_version()

        response = response.model_dump()

    res = response
    if res is None:
        res = []

    if isinstance(res, dict):
        res = response.get("predictions", response)

    return jsonify({'results': res})


@_server.route('/setup', methods=['POST'])
@exception_handler
def _setup():
    data = request.json
    project_id = data.get('project').split('.', 1)[0]
    label_config = data.get('schema')
    extra_params = data.get('extra_params')
    model.project_id = project_id
    model.use_label_config(label_config)

    if extra_params:
        model.set_extra_params(extra_params)

    model_version = model.get('model_version')
    return jsonify({'model_version': model_version})


TRAIN_EVENTS = (
    'ANNOTATION_CREATED',
    'ANNOTATION_UPDATED',
    'ANNOTATION_DELETED',
    'START_TRAINING'
)


@_server.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    event = data.pop('action')
    if event not in TRAIN_EVENTS:
        return jsonify({'status': 'Unknown event'}), 200
    project_id = str(data['project']['id'])
    label_config = data['project']['label_config']
    model.project_id = project_id
    model.use_label_config(label_config)
    model.fit(event, data)
    return jsonify({}), 201


@_server.route('/health', methods=['GET'])
@_server.route('/', methods=['GET'])
@exception_handler
def health():
    return jsonify({
        'status': 'UP',
    })


@_server.route('/metrics', methods=['GET'])
@exception_handler
def metrics():
    return jsonify({})


@_server.errorhandler(FileNotFoundError)
def file_not_found_error_handler(error):
    logger.warning('Got error: ' + str(error))
    return str(error), 404


@_server.errorhandler(AssertionError)
def assertion_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


@_server.errorhandler(IndexError)
def index_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


def safe_str_cmp(a, b):
    return hmac.compare_digest(a, b)


@_server.before_request
def check_auth():
    if BASIC_AUTH is not None:

        auth = request.authorization
        if not auth or not (safe_str_cmp(auth.username, BASIC_AUTH[0]) and safe_str_cmp(auth.password, BASIC_AUTH[1])):
            return Response('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})


@_server.before_request
def log_request_info():
    logger.debug('Request headers: %s', request.headers)
    logger.debug('Request body: %s', request.get_data())


@_server.after_request
def log_response_info(response):
    logger.debug('Response status: %s', response.status)
    logger.debug('Response headers: %s', response.headers)
    logger.debug('Response body: %s', response.get_data())
    return response
