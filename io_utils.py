from pathlib import Path


def ensure_dir(path):
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def out_path(base_dir, filename):
    if base_dir:
        ensure_dir(base_dir)
        return str(Path(base_dir) / filename)
    return filename

