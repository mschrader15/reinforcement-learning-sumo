"""
This file contains various utilities for dealing with rllib files 


"""
import os
import json5 as json


def make_directory(path):

    os.makedirs(path, exist_ok=True)


def get_rllib_pkl(path):

    from ray.cloudpickle import cloudpickle
    """Return the data from the specified rllib configuration file."""
    config_path = os.path.join(path, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(path, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError("Could not find params.pkl in either the checkpoint dir or " "its parent directory.")
    with open(config_path, 'rb') as f:
        config = cloudpickle.load(f)
    return config


def get_rllib_config(path):
    """Return the data from the specified rllib configuration file."""
    config_path = os.path.join(path, "params.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(path, "../params.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Could not find params.json in either the checkpoint dir or its parent directory. {path}")
    with open(config_path) as f:
        config = json.load(f)
    return config
