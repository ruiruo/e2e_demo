import json
from utils.decorator_train import finish, init


def get_json_content(json_file_path: str):
    try:
        with open(json_file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error reading JSON file '{json_file_path}': {e}!")


def decorator_function(train_function):
    def wrapper_function(*args, **kwargs):
        init(*args, **kwargs)
        train_function(*args, **kwargs)
        finish(*args, **kwargs)

    return wrapper_function
