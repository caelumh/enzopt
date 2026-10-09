"""YAML reading/writing. Uses PyYAML when installed, otherwise the vendored pure-Python PyYAML (KIMM-CAD Python)."""

from __future__ import annotations

import importlib
from pathlib import Path


def yaml_module(force_vendored: bool = False):
    if not force_vendored:
        try:
            return importlib.import_module("yaml")
        except ImportError:
            pass
    return importlib.import_module("enzopt._vendor.yaml")


def load_yaml_text(text: str, force_vendored: bool = False):
    return yaml_module(force_vendored).safe_load(text)


def load_yaml(path: str | Path, force_vendored: bool = False):
    with open(path, encoding="utf-8") as fh:
        return load_yaml_text(fh.read(), force_vendored)


def dump_yaml(obj, path: str | Path | None = None) -> str:
    text = yaml_module().safe_dump(obj, allow_unicode=True, sort_keys=False)
    if path is not None:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return text
