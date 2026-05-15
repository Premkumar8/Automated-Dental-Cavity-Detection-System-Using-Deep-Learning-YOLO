from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from dental_xray_utils import (
    DEFAULT_IMAGE_SIZE,
    build_datasets,
    build_model,
    compute_class_weights,
    evaluate_model,
    fine_tune_model,
    save_class_names,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dental panoramic X-ray disease classifier.")
    parser.add_argument("--dataset-dir", required=True, help="Root directory containing the downloaded dataset.")
    parser.add_argument("--epochs", type=int, default=15, help="Initial training epochs.")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="Additional fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE[0], help="Input image size.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for saved models and metrics.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(
        dataset_dir=args.dataset_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    class_weights = compute_class_weights(datasets.data_root, datasets.class_names)

    print(f"Training root: {datasets.data_root}")
    print(f"Classes: {datasets.class_names}")
    print(f"Train images: {datasets.train_count}")
    print(f"Validation images: {datasets.val_count}")
    print(f"Class weights: {class_weights}")

    model = build_model(num_classes=len(datasets.class_names), image_size=(args.image_size, args.image_size))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    initial_history = model.fit(
        datasets.train_ds,
        validation_data=datasets.val_ds,
        epochs=args.epochs,
        class_weight=class_weights if class_weights else None,
        callbacks=callbacks,
        verbose=1,
    )

    history_payload = {"initial": initial_history.history}

    if args.fine_tune_epochs > 0:
        model = fine_tune_model(model)
        fine_tune_history = model.fit(
            datasets.train_ds,
            validation_data=datasets.val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights if class_weights else None,
            callbacks=callbacks,
            verbose=1,
        )
        history_payload["fine_tune"] = fine_tune_history.history

    model.save(output_dir / "final_model.keras")
    save_class_names(datasets.class_names, output_dir / "class_names.json")

    metrics = evaluate_model(model, datasets.val_ds, datasets.class_names)
    metrics.update(
        {
            "data_root": str(datasets.data_root),
            "train_images": datasets.train_count,
            "validation_images": datasets.val_count,
            "class_names": datasets.class_names,
        }
    )

    save_json(metrics, output_dir / "metrics.json")
    save_json(history_payload, output_dir / "history.json")

    print(f"Saved final model to: {output_dir / 'final_model.keras'}")
    print(f"Saved best model to: {output_dir / 'best_model.keras'}")
    print(f"Saved labels to: {output_dir / 'class_names.json'}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
