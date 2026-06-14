from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent
PATH_KEYS = {"log_dir", "rgb_list_db", "rgb_list_q"}


def resolve_repo_path(path: str | Path, *, yaml_file: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists() or yaml_file is None:
        return repo_candidate.resolve()

    return (Path(yaml_file).expanduser().resolve().parent / candidate).resolve()


def load_exp_yaml(yaml_file: str | Path, *, resolve_paths: bool = True) -> dict[str, Any]:
    yaml_path = resolve_repo_path(yaml_file)
    with yaml_path.open("r", encoding="utf-8") as f:
        exp = yaml.safe_load(f)

    if not isinstance(exp, dict):
        raise ValueError(f"{yaml_path} must contain a YAML mapping")

    if resolve_paths:
        exp = dict(exp)
        for key in PATH_KEYS:
            if key in exp and exp[key] is not None:
                exp[key] = str(resolve_repo_path(exp[key], yaml_file=yaml_path))

    return exp
