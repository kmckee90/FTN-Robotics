"""Strict, safe loading of portable model YAMLs; experiment options stay in Python."""
from dataclasses import fields
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin, get_type_hints
import math
import yaml
from .config import ModelConfig


class UniqueSafeLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently taking the last spelling."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError('Model YAML keys must be strings')
        if key in result:
            raise ValueError(f'Duplicate model YAML key: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _matches(value, annotation):
    if get_origin(annotation) is UnionType:
        return any(_matches(value, option) for option in get_args(annotation))
    if annotation is float:
        return type(value) in (int, float) and math.isfinite(value)
    return type(value) is annotation


def load_model_yaml(path):
    """Return validated overrides and source text, without mutating a config."""
    raw = str(path)
    path = Path(path).expanduser()
    if not path.is_absolute() and '/' not in raw and '\\' not in raw:
        if not path.suffix:
            path = path.with_suffix('.yaml')
        path = Path(__file__).resolve().parent / 'presets' / path
    path = path.resolve()
    try:
        source = path.read_text()
        overrides = yaml.load(source, Loader=UniqueSafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f'Cannot load model config {path}: {exc}') from exc
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError('Model YAML must be a nonempty mapping')
    if 'model' not in overrides:
        raise ValueError('Model YAML must specify model')
    allowed = {field.name for field in fields(ModelConfig)}
    unknown = overrides.keys() - allowed
    if unknown:
        raise ValueError(f'Unknown model YAML fields: {", ".join(sorted(unknown))}. '
                         'Task/data/training/output options belong to the experiment CLI.')
    hints = get_type_hints(ModelConfig)
    for key, value in overrides.items():
        if not _matches(value, hints[key]):
            raise ValueError(f'Invalid type for model YAML field {key}: expected {hints[key]}, got {value!r}')
    return overrides, source
