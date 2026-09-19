"""Subject-class constraints per predicate."""
SUBJECT_CLASSES = {
    "holding":     {"person"},
    "riding":      {"person"},
    "wearing":     {"person"},
    "sitting on":  {"person"},
    "walking on":  {"person"},
    "walking past":{"person"},
    "looking at":  {"person"},
    "picking up":  {"person"},
    "putting down":{"person"},
    "carrying":    {"person"},
    "using":       {"person"},
}

def is_compatible(subject_id: str, predicate: str) -> bool:
    allowed = SUBJECT_CLASSES.get(predicate.lower())
    if not allowed:
        return True
    return subject_id.rsplit("_", 1)[0] in allowed
