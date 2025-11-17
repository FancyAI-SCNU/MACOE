import yaml
import re
import os
import copy
from util import deep_update

loader = yaml.FullLoader
loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)

def merge_dicts(d1, d2):
    """

    :param d1: Dict 1.
    :type d1: dict
    :param d2: Dict 2.
    :returns: A new dict that is d1 and d2 deep merged.
    :rtype: dict

    """
    merged = copy.deepcopy(d1)
    deep_update(merged, d2, True, [])
    return merged

def get_full_config(config, dir_name):
    while "base" in config:
        base_config = os.path.normpath(os.path.join(dir_name, config.pop("base")))
        dir_name = os.path.dirname(base_config)
        with open(base_config, "r") as f:
            base_config = yaml.load(base_config, Loader=yaml.FullLoader)
        config = merge_dicts(base_config, config)
    return config

def return_config():
    with open("config.yml", "r") as f:
        config = yaml.load(f, Loader=loader)
    config = get_full_config(config, os.path.dirname("config.yml"))
    print(config)
    return config

