import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="nbaxp")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage", help="raw snapshot -> data/staged")
    sub.add_parser("features", help="data/staged -> data/features")
    args = parser.parse_args()

    t0 = time.time()
    if args.command == "stage":
        from nbaxp.data.stage import stage

        for name, n in stage(args.data_dir).items():
            print(f"staged {name}: {n} rows")
    elif args.command == "features":
        from nbaxp.features.build import build

        print(f"features: {build(args.data_dir)} rows")
    print(f"{args.command} took {time.time() - t0:.1f} s")
