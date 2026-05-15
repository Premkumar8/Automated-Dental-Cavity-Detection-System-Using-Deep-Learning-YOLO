from __future__ import annotations

import argparse
import time

import kagglehub


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle dental panoramic disease dataset.")
    parser.add_argument(
        "--dataset",
        default="lokisilvres/dental-disease-panoramic-detection-dataset",
        help="Kaggle dataset slug.",
    )
    parser.add_argument("--retries", type=int, default=3, help="Number of download attempts.")
    parser.add_argument("--sleep-seconds", type=int, default=10, help="Sleep between retries.")
    args = parser.parse_args()

    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            path = kagglehub.dataset_download(args.dataset)
            print(f"Dataset downloaded to: {path}")
            return
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"Attempt {attempt}/{args.retries} failed: {exc}")
            if attempt < args.retries:
                time.sleep(args.sleep_seconds)

    raise SystemExit(f"Dataset download failed after {args.retries} attempts: {last_error}")


if __name__ == "__main__":
    main()
