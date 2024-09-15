import json
import os

class Config:
    """A simple class for accessing config

    This class allows config to be access like an object
    instead of accessing through a dictionary.
    
    Examples
    --------
    >>> cfg_dict = {
    ...     data_path = '/path/to/data',
    ...     models_dir = '/models/dir'
    ... } 
    >>> cfg = Config(cfg_dict)
    >>> cfg.data_path
    '/path/to/data'
    >>> cfg.models_dir
    '/models/dir'
    """

    def __init__(self, entries):
        self.__dict__.update(**entries)

def read_config(cfg_path: str | os.PathLike) -> Config:
    """A function to read script configuration from a .json file.

    Parameters
    ----------
    cfg_path : str | os.PathLike
        The path of the json.file
    
    Return
    ------
    utils.Config
        A Config object containing the configuration.
    """

    with open(cfg_path, 'r') as file:
        cfg_dict = json.load(file)
    cfg = Config(cfg_dict)
    return cfg