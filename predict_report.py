from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from dental_xray_utils import generate_report, load_and_prepare_image, load_class_names, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict dental disease from one panoramic X-ray and generate a report.")
    parser.add_argument("--image", required=True, help="Path to the X-ray image.")
    parser.add_argument("--model", default="artifacts/best_model.keras", help="Path to the trained Keras model.")
    parser.add_argument("--labels", default="artifacts/class_names.json", help="Path to class_names.json.")
    parser.add_argument("--output", default="artifacts/latest_report.json", help="Path to save the report JSON.")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = tf.keras.models.load_model(args.model)
    class_names = load_class_names(args.labels)
    image_batch = load_and_prepare_image(image_path)
    probabilities = model.predict(image_batch, verbose=0)[0]
    report = generate_report(image_path=image_path, class_names=class_names, probabilities=probabilities)
    save_json(report, args.output)

    print(f"Image: {report['image']}")
    print(f"Prediction: {report['predicted_label']}")
    print(f"Confidence: {report['predicted_confidence']:.2%}")
    print("Top predictions:")
    for item in report["top_predictions"]:
        print(f"  - {item['label']}: {item['confidence']:.2%}")
    print("Report:")
    print(report["report"])
    print(f"JSON saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
