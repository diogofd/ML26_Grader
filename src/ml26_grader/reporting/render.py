from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def render_json_document(payload: BaseModel | dict[str, Any]) -> str:
    if isinstance(payload, BaseModel):
        return payload.model_dump_json(indent=2)
    return json.dumps(payload, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return str(value)
