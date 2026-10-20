"""Cloud data sync. With DATA_ROOT=gs://bucket[/prefix], a command downloads its inputs into a local
working dir before it runs and uploads its outputs after. The folder layout is the same locally and in the bucket."""

import os
from pathlib import Path

WORK_DIR = Path("/tmp/nbaxp-data")
STAGE_TABLES = ("games", "teams", "player_stats")  # the only raw tables `stage` reads

# command -> folders to download before / upload after
INPUTS = {
    "stage": ("raw",),
    "features": ("staged",),
    "train": ("features", "mlflow"),
    "replay": ("features", "mlflow", "reports"),
    "pipeline": ("raw", "mlflow", "reports"),  # mlflow and reports tell the pipeline whether it can skip train
}
OUTPUTS = {
    "stage": ("staged",),
    "features": ("features",),
    "train": ("mlflow", "reports"),
    "replay": ("mlflow", "reports", "replay"),
    "pipeline": ("staged", "features", "mlflow", "reports", "replay"),
}


def split_root(root: str) -> tuple[str, str]:
    """'gs://bucket/some/prefix' -> ('bucket', 'some/prefix')."""
    bucket, _, prefix = root.removeprefix("gs://").partition("/")
    return bucket, prefix.strip("/")


def key(prefix: str, folder: str) -> str:
    return f"{prefix}/{folder}" if prefix else folder


def latest_raw(names: list[str]) -> list[str]:
    """Keep the blobs of the newest snapshot and of the tables `stage` reads. Names look like raw/snapshot=X/table/file."""
    parts = [n.split("/") for n in names if not n.endswith("/")]
    newest = max(p[1] for p in parts)
    return ["/".join(p) for p in parts if p[1] == newest and p[2] in STAGE_TABLES]


def download(root: str, local: Path, folders, client=None) -> None:
    from google.cloud import storage

    bucket_name, prefix = split_root(root)
    bucket = (client or storage.Client()).bucket(bucket_name)
    strip = len(prefix) + 1 if prefix else 0
    for folder in folders:
        blobs = {b.name[strip:]: b for b in bucket.list_blobs(prefix=key(prefix, folder) + "/")}
        names = latest_raw(list(blobs)) if folder == "raw" and blobs else [n for n in blobs if not n.endswith("/")]
        for name in names:
            path = local / name
            path.parent.mkdir(parents=True, exist_ok=True)
            blobs[name].download_to_filename(str(path))


def upload(root: str, local: Path, folders, client=None) -> None:
    from google.cloud import storage

    bucket_name, prefix = split_root(root)
    bucket = (client or storage.Client()).bucket(bucket_name)
    for folder in folders:
        for path in sorted((local / folder).rglob("*")):
            if path.is_file():
                bucket.blob(key(prefix, path.relative_to(local).as_posix())).upload_from_filename(str(path))


def data_root() -> str | None:
    root = os.environ.get("DATA_ROOT", "")
    return root if root.startswith("gs://") else None
