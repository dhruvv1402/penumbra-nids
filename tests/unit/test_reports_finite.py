"""Reports with undefined values are served as JSON null, not as bare NaN that breaks the route."""

from __future__ import annotations

import json
import math

from penumbra.api.reports import finite


def test_non_finite_values_become_null_at_any_depth() -> None:
    raw = {"a": math.nan, "b": [1.0, math.inf, {"c": -math.inf}], "d": "NaN", "e": 2}
    cleaned = finite(raw)
    assert cleaned == {"a": None, "b": [1.0, None, {"c": None}], "d": "NaN", "e": 2}
    json.dumps(cleaned, allow_nan=False)  # what Starlette does; must not raise
