"""
This file contains various utilities for dealing with rllib files 


"""
import os
from typing import OrderedDict
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


def read_nema_config(path: str, tlsID: str = "") -> OrderedDict:
    """
    Reads a SUMO NEMA configuration file. 

    Description here: https://sumo.dlr.de/docs/Simulation/NEMA.html

    Args:
        path (str): Path to the sumo add.xml file describing the NEMA traffic light
        tlsID (str, optional): The tlsID, useful if there are multiple traffic lights described in the same file. Defaults to "".

    Returns:
        OrderedDict: _description_
    """
    import xmltodict
    from collections import OrderedDict


    with open(path, "r") as f:
        raw = xmltodict.parse(f.read())
        # get passed the first element (either called "add" or "additional")
        tl_dict = raw[next(iter(raw.keys()))]["tlLogic"]

    # If there are multiple traffic lights in the file get the one we want
    if isinstance(tl_dict, list):
        assert(len(tlsID))
        tl_dict = [o for o in tl_dict if o['@programID'] == tlsID][0]

    # turn all the "params" into a unique dictionary with the key being the "Key" and "Value" being the "value"
    tl_dict['param'] = {
        p["@key"] : p["@value"] for p in tl_dict['param']
    }

    # find "skip" phases. These are phases that are always green.
    skip_phases = []
    for i, p in enumerate(tl_dict['phase'][:-1]):
        skip_phases.extend(iid for iid, (_p, _pnext) in enumerate(zip(p["@state"], tl_dict['phase'][i + 1]["@state"])) if _p == "G" and _pnext == "G")
    skip_phases = set(skip_phases)

    # skip_phases = [iid for i, p in enumerate(tl_dict['phase'][:-1]) for iid, (_p, _pnext) in enumerate(zip(p["@state"], tl_dict['phase'][i+1]["@state"])) if _p == _pnext]


    # add a parameter called the controlling index, useful for mapping phase names to lanes
    # this might not always be right, but can't think of immediately better way to do it
    for phase in tl_dict['phase']:
        phase['controlling_index'] = [i for i, s in enumerate(phase["@state"]) if s == "G" and i not in skip_phases]

    # CRITICAL, sort the phases by their controlling indexs
    tl_dict['phase'] = OrderedDict(
        p[:-1] for p in sorted(((phase["@name"], phase, min(phase['controlling_index'])) for phase in tl_dict['phase']), key=lambda x: x[-1])
    )

    return tl_dict

