"""Export a frozen snapshot of the source BigQuery tables to local Parquet.

Reads each table whole through the BigQuery Storage Read API (no query is run),
streams it to Parquet part files, and writes a manifest with row counts, source
table metadata and file hashes. The snapshot is the only data source for the
rest of the pipeline.

Usage:
    gcloud auth application-default login
    python scripts/export_snapshot.py --project P --dataset D                 # all default tables
    python scripts/export_snapshot.py --project P --dataset D --tables games  # one table
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from google.cloud import bigquery, bigquery_storage

DEFAULT_TABLES = ["games", "players", "teams", "player_stats", "shots", "playbyplay"]
ROWS_PER_FILE = 2_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_table(bq, bqs, table_id: str, out_dir: Path) -> dict:
    table = bq.get_table(table_id)
    print(f"{table_id}: {table.num_rows:,} rows, {table.num_bytes / 1e9:.2f} GB in BigQuery")

    # Write to a temp dir and rename at the end, so a crash never leaves a
    # half-written table that looks complete.
    tmp_dir = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    writer, part, rows_in_part, rows_total = None, 0, 0, 0
    batches = bq.list_rows(table).to_arrow_iterable(bqstorage_client=bqs)
    for batch in batches:
        if writer is None or rows_in_part >= ROWS_PER_FILE:
            if writer is not None:
                writer.close()
                part += 1
            writer = pq.ParquetWriter(tmp_dir / f"part-{part:05d}.parquet", batch.schema, compression="zstd")
            rows_in_part = 0
        writer.write_batch(batch)
        rows_in_part += batch.num_rows
        rows_total += batch.num_rows
        print(f"\r  {rows_total:,} / {table.num_rows:,}", end="", flush=True)
    if writer is not None:
        writer.close()
    print()

    files = sorted(tmp_dir.glob("*.parquet"))
    parquet_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if parquet_rows != table.num_rows:
        raise RuntimeError(f"{table_id}: Parquet has {parquet_rows:,} rows, BigQuery reports {table.num_rows:,}")

    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    files = sorted(out_dir.glob("*.parquet"))

    return {
        "source": table_id,
        "source_last_modified": table.modified.isoformat(),
        "source_num_rows": table.num_rows,
        "source_num_bytes": table.num_bytes,
        "parquet_rows": parquet_rows,
        "schema": [{"name": f.name, "type": f.field_type, "mode": f.mode} for f in table.schema],
        "files": [
            {"path": f"{out_dir.name}/{f.name}", "bytes": f.stat().st_size, "sha256": sha256(f)}
            for f in files
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True, help="project that owns the dataset and is billed for the read")
    p.add_argument("--dataset", required=True)
    p.add_argument("--tables", nargs="+", default=DEFAULT_TABLES)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--snapshot", default=date.today().isoformat(), help="snapshot label, defaults to today")
    p.add_argument("--force", action="store_true", help="overwrite tables already in this snapshot")
    args = p.parse_args()

    snap_dir = args.out / f"snapshot={args.snapshot}"
    manifest_path = snap_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"tables": {}}

    bq = bigquery.Client(project=args.project)
    bqs = bigquery_storage.BigQueryReadClient()

    for name in args.tables:
        out_dir = snap_dir / name
        if out_dir.exists() and not args.force:
            print(f"{name}: already exported, skipping (use --force to redo)")
            continue
        manifest["tables"][name] = export_table(bq, bqs, f"{args.project}.{args.dataset}.{name}", out_dir)
        # Rewrite after every table so an interrupted run keeps what finished.
        manifest.update(snapshot=args.snapshot, exported_at=datetime.now(timezone.utc).isoformat())
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
