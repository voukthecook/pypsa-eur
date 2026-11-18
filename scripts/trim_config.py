import subprocess, yaml

def diff(base, new):
    out = {}
    for k, v in new.items():
        if k not in base:
            out[k] = v
        elif isinstance(v, dict) and isinstance(base[k], dict):
            d = diff(base[k], v)
            if d:
                out[k] = d
        elif v != base[k]:
            out[k] = v
    return out

default = yaml.safe_load(subprocess.check_output(
    ["git", "show", "HEAD:config/config.default.yaml"]))
mine = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))

with open("config/config.minimal.yaml", "w", encoding="utf-8") as f:
    yaml.dump(diff(default, mine), f, sort_keys=False, allow_unicode=True)
print(open("config/config.minimal.yaml", encoding="utf-8").read())