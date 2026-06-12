from __future__ import annotations

import argparse
import json

from .api import hide, reveal
from .collect import collect
from .eval_suite import run_eval_suite
from .experiment import demo, run_pairs, run_trends
from .io import load_rgb, save_rgb
from .report_plots import make_report_plots


def main() -> None:
    parser = argparse.ArgumentParser(prog="sfvmosaic")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("hide")
    p.add_argument("secret")
    p.add_argument("target")
    p.add_argument("output")
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--mode", choices=("robust", "paper"), default="robust")
    p = commands.add_parser("extract")
    p.add_argument("mosaic")
    p.add_argument("output")
    p.add_argument("--key", required=True)
    p = commands.add_parser("demo")
    p.add_argument("--secret", default="secret.png")
    p.add_argument("--target", default="target.png")
    p.add_argument("--output", default="outputs/demo")
    p.add_argument("--block", type=int, default=37)
    p.add_argument("--key", default="paper-demo-key")
    p.add_argument("--mode", choices=("robust", "paper"), default="robust")
    p = commands.add_parser("collect-data")
    p.add_argument("--output", default="data/kodak")
    p.add_argument("--count", type=int, default=12)
    p = commands.add_parser("experiment")
    p.add_argument("--secret", default="secret.png")
    p.add_argument("--target", default="target.png")
    p.add_argument("--output", default="outputs/experiment")
    p.add_argument("--blocks", default="8,16,32")
    p.add_argument("--pairs-dataset")
    p.add_argument("--mode", choices=("robust", "paper"), default="paper")
    p.add_argument("--embedded", action="store_true")
    p = commands.add_parser("eval-suite")
    p.add_argument("--secret", default="secret.png")
    p.add_argument("--target", default="target.png")
    p.add_argument("--output", default="outputs/eval_suite")
    p.add_argument("--blocks", default="37")
    p.add_argument("--key", default="eval-suite-key")
    p.add_argument("--no-embedded", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p = commands.add_parser("plot-report")
    p.add_argument("--experiment-dir", default="outputs/experiment")
    p.add_argument("--eval-dir", default="outputs/eval_suite")
    p.add_argument("--output", default="outputs/report_figures")
    args = parser.parse_args()

    if args.command == "hide":
        image, stats = hide(
            load_rgb(args.secret), load_rgb(args.target), args.block, args.key, mode=args.mode
        )
        save_rgb(args.output, image, rgba=True)
        print(json.dumps(stats, indent=2))
    elif args.command == "extract":
        image, stats = reveal(load_rgb(args.mosaic), args.key)
        save_rgb(args.output, image, rgba=True)
        print(json.dumps(stats, indent=2))
    elif args.command == "demo":
        print(
            json.dumps(
                demo(args.secret, args.target, args.output, args.block, args.key, args.mode),
                indent=2,
            )
        )
    elif args.command == "plot-report":
        print(
            json.dumps(
                make_report_plots(args.experiment_dir, args.eval_dir, args.output),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "collect-data":
        print(json.dumps(collect(args.output, args.count), indent=2))
    elif args.command == "experiment":
        result = {
            "trends": run_trends(
                args.secret,
                args.target,
                args.output,
                [int(value) for value in args.blocks.split(",")],
                mode=args.mode,
                include_embedded=args.embedded,
            )
        }
        if args.pairs_dataset:
            result["pairs"] = len(
                run_pairs(
                    args.pairs_dataset,
                    args.output,
                    [int(value) for value in args.blocks.split(",")],
                    mode=args.mode,
                    include_embedded=args.embedded,
                )
            )
        print(json.dumps(result, indent=2))
    elif args.command == "eval-suite":
        print(
            json.dumps(
                run_eval_suite(
                    args.secret,
                    args.target,
                    args.output,
                    [int(value) for value in args.blocks.split(",")],
                    embedded=not args.no_embedded,
                    key=args.key,
                    seed=args.seed,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
