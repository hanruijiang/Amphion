# Copyright (c) 2023 Amphion. MIT license; see LICENSE.
import json5


def load_config(config_fn):
    """Load a self-contained published model configuration (JSON5)."""
    with open(config_fn, "r", encoding="utf-8") as f:
        config = json5.load(f)
    if "base_config" in config:
        raise ValueError("Inherited model configurations are not supported")
    return JsonHParams(**config)


class JsonHParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


