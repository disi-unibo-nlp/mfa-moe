"""Canonical unavailable values at every data, hashing and publication boundary."""
import math
from collections.abc import Mapping
import numbers


def normalize(value, reasons=None, path=""):
    if isinstance(value, Mapping):
        return {str(k): normalize(v, reasons, path+"/"+str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v, reasons, path+"/"+str(i)) for i, v in enumerate(value)]
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        if reasons is not None: reasons[path] = "missing_input"
        return None
    if isinstance(value, numbers.Real):
        if not math.isfinite(value):
            if reasons is not None: reasons[path] = "non_finite_input"
            return None
        return value.item() if hasattr(value, "item") else value
    if hasattr(value, "tolist"):
        return normalize(value.tolist(), reasons, path)
    return value


def records(rows):
    output = []
    for row in rows:
        reasons = {}
        clean = normalize(row, reasons)
        if reasons:
            clean["unavailable_reasons"] = {**clean.get("unavailable_reasons", {}), **reasons}
        output.append(clean)
    return output


def read_parquet(path):
    import pandas as pd
    return records(pd.read_parquet(path).to_dict("records"))


def finite(value):
    return isinstance(value, numbers.Real) and math.isfinite(value)
