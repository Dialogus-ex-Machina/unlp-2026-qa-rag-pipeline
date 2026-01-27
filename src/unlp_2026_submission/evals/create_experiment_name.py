import re

def create_experiment_name(**kwargs) -> str:
    def clean(v: object) -> str:
        s = str(v)
        s = re.sub(r"\s+", "-", s)
        s = re.sub(r"[^a-zA-Z0-9._-]", "", s)
        return s

    return "_".join(
        clean(kwargs[k])
        for k in sorted(kwargs)
        if kwargs[k] is not None
    ) + "_experiment"