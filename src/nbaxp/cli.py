import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="nbaxp")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage", help="raw snapshot -> data/staged")
    sub.add_parser("features", help="data/staged -> data/features")
    sub.add_parser("train", help="data/features -> dev CV, MLflow, data/reports/dev_cv.json")
    args = parser.parse_args()

    t0 = time.time()
    if args.command == "stage":
        from nbaxp.data.stage import stage

        for name, n in stage(args.data_dir).items():
            print(f"staged {name}: {n} rows")
    elif args.command == "features":
        from nbaxp.features.build import build

        print(f"features: {build(args.data_dir)} rows")
    elif args.command == "train":
        from nbaxp.model.tracking import train

        report = train(args.data_dir)
        print(f"selected {report['selected_params']}, sigma {report['sigma']:.3f}")
        print(f"registered version {report['registered_version']} in {args.data_dir / 'mlflow'}")
    print(f"{args.command} took {time.time() - t0:.1f} s")
