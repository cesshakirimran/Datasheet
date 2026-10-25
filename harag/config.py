"""Configuration loading: YAML file + optional `inherit:` base file + ${ENV_VAR} expansion."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


class Settings(dict):
    """A dict with attribute access and a few path helpers.

    Paths in the config are resolved relative to the project root (the folder of the config file).
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc
        return Settings(value) if isinstance(value, dict) and not isinstance(value, Settings) else value

    @property
    def root(self) -> Path:
        return Path(self["_root"])

    def path(self, key: str) -> Path:
        p = Path(self[key])
        return p if p.is_absolute() else self.root / p


def load_settings(path: str | os.PathLike = "config.yaml") -> Settings:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("inherit", None)
    if parent:
        base = load_settings(path.parent / parent)
        base.pop("_root", None)
        data = _deep_merge(dict(base), data)
    data = _expand_env(data)
    data["_root"] = str(path.parent)
    data["_config_file"] = str(path)
    return Settings(data)
