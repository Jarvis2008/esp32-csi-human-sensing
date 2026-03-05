"""Export labeled session windows into a training CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from host.inference.dataset import TRAINING_LABELS, write_labeled_dataset_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Export labeled CSI windows to CSV")
    parser.add_argument("--session-root", default="data/sessions")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--include-unlabeled", action="store_true")
    parser.add_argument(
        "--labels",
        default=",".join(TRAINING_LABELS),
        help="comma-separated labels to keep (default: training labels)",
    )
    args = parser.parse_args()

    labels = tuple(sorted({token.strip() for token in args.labels.split(",") if token.strip()}))
    result = write_labeled_dataset_csv(
        session_root=Path(args.session_root),
        output_csv=Path(args.output_csv),
        include_unlabeled=args.include_unlabeled,
        allowed_labels=labels,
    )

    print(f"Output CSV: {result.output_csv}")
    print(f"Feature columns: {len(result.feature_names)}")
    print(f"Rows total: {result.total_rows}")
    print(f"Rows kept: {result.kept_rows}")
    print(f"Rows skipped (unlabeled): {result.skipped_unlabeled}")
    print(f"Rows skipped (unknown): {result.skipped_unknown_label}")
    print("Label counts:")
    for label, count in sorted(result.label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
