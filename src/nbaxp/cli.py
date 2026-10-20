import argparse
import os
import time
from pathlib import Path

from nbaxp import sync


def main() -> None:
    parser = argparse.ArgumentParser(prog="nbaxp")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage", help="raw snapshot -> data/staged")
    sub.add_parser("features", help="data/staged -> data/features")
    sub.add_parser("train", help="data/features -> dev CV, MLflow, data/reports/dev_cv.json")
    sub.add_parser("replay", help="replay 2025-26 day by day -> data/replay, data/reports/replay.json")
    sub.add_parser("pipeline", help="stage, features, train (if no champion yet), replay")
    sub.add_parser("api", help="serve the 2025-26 predictions on 0.0.0.0:$PORT (default 8080)")
    args = parser.parse_args()

    root = sync.data_root()  # DATA_ROOT=gs://... : work in a local dir, sync with the bucket
    if root:
        args.data_dir = sync.WORK_DIR
        sync.download(root, args.data_dir, sync.INPUTS[args.command])
    t0 = time.time()
    if args.command in ("stage", "pipeline"):
        from nbaxp.data.stage import stage

        for name, n in stage(args.data_dir).items():
            print(f"staged {name}: {n} rows")
    if args.command in ("features", "pipeline"):
        from nbaxp.features.build import build

        print(f"features: {build(args.data_dir)} rows")
    if args.command in ("train", "pipeline"):
        from nbaxp.model.tracking import has_champion, train

        if args.command == "train" or not has_champion(args.data_dir):
            report = train(args.data_dir)
            print(f"selected {report['selected_params']}, sigma {report['sigma']:.3f}")
            print(f"registered version {report['registered_version']} in {args.data_dir / 'mlflow'}")
    if args.command in ("replay", "pipeline"):
        from nbaxp.replay import replay

        s = replay(args.data_dir)
        print(f"replay: {s['n_games']} games, {s['n_checks']} checks, {s['n_promoted']} promoted")
        print(f"MAE chain {s['champion_chain']['mae']:.3f}, frozen v1 {s['frozen_v1']['mae']:.3f}")
    if args.command == "api":
        import uvicorn

        from nbaxp.api import create_app

        uvicorn.run(create_app(args.data_dir), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
    print(f"{args.command} took {time.time() - t0:.1f} s")
    if root:
        sync.upload(root, args.data_dir, sync.OUTPUTS[args.command])
