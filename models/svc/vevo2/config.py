# Copyright (c) 2023 Amphion. MIT license; see LICENSE.
import json5


class JsonHParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, JsonHParams(**value) if isinstance(value, dict) else value)


def load_config(path):
    with open(path, encoding="utf-8") as file:
        return JsonHParams(**json5.load(file))
