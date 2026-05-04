from __future__ import annotations

import argparse
from pathlib import Path

from flygen_ml.modeling.joint_evaluation import (
    join_prediction_rows,
    load_prediction_rows,
    prediction_path_for_input,
    summarize_joint_predictions,
    write_joint_prediction_rows,
)


def _format_metric(value: object) -> str:
    return f"{float(value):.3f}"


def _print_confusion_matrix(summary: dict[str, object]) -> None:
    labels = [str(label) for label in list(summary["joint_labels"])]
    matrix = dict(summary["joint_confusion_matrix"])
    print("joint_confusion_matrix:")
    print("actual\\predicted\t" + "\t".join(labels))
    for actual in labels:
        row = dict(matrix[actual])
        print(f"{actual}\t" + "\t".join(str(row[predicted]) for predicted in labels))


def _print_summary(summary: dict[str, object], *, axis_a_name: str, axis_b_name: str) -> None:
    print(f"n_joined_examples: {int(summary['n_joined_examples'])}")
    print(f"joint_accuracy: {_format_metric(summary['joint_accuracy'])}")
    print(f"{axis_a_name}_accuracy: {_format_metric(summary['axis_a_accuracy'])}")
    print(f"{axis_b_name}_accuracy: {_format_metric(summary['axis_b_accuracy'])}")
    print(
        "correctness_counts: "
        f"both_correct={int(summary['both_correct'])}, "
        f"{axis_a_name}_only_wrong={int(summary['axis_a_only_wrong'])}, "
        f"{axis_b_name}_only_wrong={int(summary['axis_b_only_wrong'])}, "
        f"both_wrong={int(summary['both_wrong'])}"
    )
    print()
    _print_confusion_matrix(summary)

    by_evidence_bin = dict(summary.get("by_evidence_bin", {}))
    if by_evidence_bin:
        print()
        print("by_evidence_bin:")
        for evidence_bin, payload in sorted(by_evidence_bin.items()):
            values = dict(payload)
            print(
                f"{evidence_bin}: "
                f"n={int(values['n_joined_examples'])}, "
                f"joint_accuracy={_format_metric(values['joint_accuracy'])}, "
                f"{axis_a_name}_accuracy={_format_metric(values['axis_a_accuracy'])}, "
                f"{axis_b_name}_accuracy={_format_metric(values['axis_b_accuracy'])}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate whether two fly-level classifiers jointly identify each fly."
    )
    axis_a = parser.add_mutually_exclusive_group(required=True)
    axis_a.add_argument("--axis-a-run", help="Run directory for the first label axis.")
    axis_a.add_argument("--axis-a-predictions", help="Prediction CSV for the first label axis.")
    axis_b = parser.add_mutually_exclusive_group(required=True)
    axis_b.add_argument("--axis-b-run", help="Run directory for the second label axis.")
    axis_b.add_argument("--axis-b-predictions", help="Prediction CSV for the second label axis.")
    parser.add_argument("--axis-a-name", required=True, help="Name for the first label axis.")
    parser.add_argument("--axis-b-name", required=True, help="Name for the second label axis.")
    parser.add_argument(
        "--split",
        default="valid",
        help="Prediction split to evaluate. Use 'all' to include all splits.",
    )
    parser.add_argument(
        "--join-without-fold",
        action="store_true",
        help=(
            "Ignore fold when joining CV prediction tables. Use this for out-of-fold "
            "valid predictions from runs whose fold assignments are not aligned."
        ),
    )
    parser.add_argument("--output", help="Path for the joined prediction CSV.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    axis_a_input = args.axis_a_run or args.axis_a_predictions
    axis_b_input = args.axis_b_run or args.axis_b_predictions
    assert axis_a_input is not None
    assert axis_b_input is not None

    split = None if args.split == "all" else str(args.split)
    axis_a_path = prediction_path_for_input(axis_a_input)
    axis_b_path = prediction_path_for_input(axis_b_input)
    axis_a_rows = load_prediction_rows(axis_a_path, split=split)
    axis_b_rows = load_prediction_rows(axis_b_path, split=split)
    joined_rows, key_columns = join_prediction_rows(
        axis_a_rows,
        axis_b_rows,
        axis_a_name=args.axis_a_name,
        axis_b_name=args.axis_b_name,
        include_fold=not args.join_without_fold,
    )
    summary = summarize_joint_predictions(
        joined_rows,
        axis_a_name=args.axis_a_name,
        axis_b_name=args.axis_b_name,
    )

    print(f"axis_a_predictions: {axis_a_path}")
    print(f"axis_b_predictions: {axis_b_path}")
    print(f"join_columns: {', '.join(key_columns)}")
    if split is not None:
        print(f"prediction_split: {split}")
    print()
    _print_summary(summary, axis_a_name=args.axis_a_name, axis_b_name=args.axis_b_name)

    if args.output:
        output_path = Path(args.output)
        write_joint_prediction_rows(
            output_path,
            joined_rows,
            axis_a_name=args.axis_a_name,
            axis_b_name=args.axis_b_name,
        )
        print()
        print(f"wrote_joined_predictions: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
