"""Cloud sync tests with a fake bucket: no network."""

from pathlib import Path

from nbaxp import sync


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def download_to_filename(self, path):
        Path(path).write_text(self.bucket.files[self.name])

    def upload_from_filename(self, path):
        self.bucket.files[self.name] = Path(path).read_text()


class FakeBucket:
    def __init__(self, files):
        self.files = files

    def list_blobs(self, prefix):
        return [FakeBlob(self, n) for n in sorted(self.files) if n.startswith(prefix)]

    def blob(self, name):
        return FakeBlob(self, name)


class FakeClient:
    def __init__(self, files):
        self.fake = FakeBucket(files)

    def bucket(self, name):
        assert name == "nba-data"
        return self.fake


def test_split_root():
    assert sync.split_root("gs://nba-data") == ("nba-data", "")
    assert sync.split_root("gs://nba-data/") == ("nba-data", "")
    assert sync.split_root("gs://nba-data/run1/") == ("nba-data", "run1")


def test_data_root(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    assert sync.data_root() is None
    monkeypatch.setenv("DATA_ROOT", "data")
    assert sync.data_root() is None
    monkeypatch.setenv("DATA_ROOT", "gs://nba-data")
    assert sync.data_root() == "gs://nba-data"


def test_download_takes_newest_snapshot_and_needed_tables_only(tmp_path):
    names = [f"raw/snapshot={s}/{t}/part-0.parquet" for s in ("2026-10-20", "2026-09-01") for t in ("games", "shots")]
    names += ["raw/snapshot=2026-10-20/teams/part-0.parquet", "features/features.parquet", "mlflow/mlflow.db"]
    client = FakeClient({n: "x" for n in names})
    sync.download("gs://nba-data", tmp_path, ["raw", "features"], client=client)
    got = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert got == [
        "features/features.parquet",
        "raw/snapshot=2026-10-20/games/part-0.parquet",
        "raw/snapshot=2026-10-20/teams/part-0.parquet",
    ]


def test_upload_and_prefix_mapping(tmp_path):
    (tmp_path / "replay").mkdir()
    (tmp_path / "replay" / "daily.parquet").write_text("d")
    (tmp_path / "staged").mkdir()
    (tmp_path / "staged" / "games.parquet").write_text("g")
    client = FakeClient({})
    sync.upload("gs://nba-data/run1", tmp_path, ["replay"], client=client)
    assert client.fake.files == {"run1/replay/daily.parquet": "d"}
    # a prefixed root downloads back to the same relative layout
    sync.download("gs://nba-data/run1", tmp_path / "back", ["replay"], client=client)
    assert (tmp_path / "back" / "replay" / "daily.parquet").read_text() == "d"
