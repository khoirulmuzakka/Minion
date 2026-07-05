import argparse
import json
import os
import sys


def collect_optima(dimensions):
    try:
        import cocoex  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "cocoex is not installed in this environment. Install cocoex first, then rerun this script."
        ) from exc

    payload = {}
    for dim in dimensions:
        values = []
        for function_number in range(1, 25):
            problem = cocoex.BareProblem(
                suite_name="bbob",
                function=function_number,
                dimension=dim,
                instance=1,
            )
            values.append(float(problem.best_value()))
        payload[str(dim)] = values
    return payload


def main():
    parser = argparse.ArgumentParser(description="Dump BBOB2009 global optima to a JSON cache.")
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[2, 3, 5, 10, 20, 40],
        help="BBOB2009 dimensions to query.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "bbob2009_optima.json"),
        help="Output JSON file.",
    )
    args = parser.parse_args()

    payload = collect_optima(args.dimensions)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print(f"Wrote BBOB2009 optima cache to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
