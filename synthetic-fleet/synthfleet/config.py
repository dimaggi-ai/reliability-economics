import copy, os, yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

class Calib:
    """Calibration table: every parameter has value, unit, status, source."""
    def __init__(self, path=None):
        self.path = path or os.path.join(HERE, "configs", "calibration.yaml")
        d = load_yaml(self.path)
        self.sources = d["sources"]
        self.params = d["params"]
    def __getitem__(self, k):
        return self.params[k]["value"]
    def rows(self):
        for k, p in self.params.items():
            src = self.sources.get(p.get("source", ""), {})
            yield dict(param=k, value=p["value"], unit=p.get("unit", ""), status=p.get("status", ""),
                       source=src.get("title", p.get("source", "")), url=src.get("url", ""), note=p.get("note", ""))

def load_config(path, overrides=None):
    cfg = load_yaml(path)
    for k, v in (overrides or {}).items():
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return cfg

def load_library(path=None):
    path = path or os.path.join(HERE, "configs", "scenarios", "library.yaml")
    return load_yaml(path)["library"]
