from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", BASE_DIR / "runs" / "obb" / "cavity_obb" / "weights" / "best.pt"))
FALLBACK_MODEL_PATH = BASE_DIR / "runs" / "obb" / "cavity_obb_test" / "weights" / "best.pt"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

_MODEL = None


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def resolve_model_path() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    if FALLBACK_MODEL_PATH.exists():
        return FALLBACK_MODEL_PATH
    raise FileNotFoundError(
        f"YOLO model not found at {MODEL_PATH}. Train it first with yolo obb train."
    )


def get_model() -> YOLO:
    global _MODEL

    if _MODEL is None:
        _MODEL = YOLO(str(resolve_model_path()))

    return _MODEL


def build_detection_report(image_path: Path) -> dict:
    model = get_model()
    results = model.predict(source=str(image_path), imgsz=640, conf=0.25, verbose=False)
    result = results[0]
    names = result.names
    obb = result.obb

    detections = []
    if obb is not None and obb.cls is not None and len(obb.cls) > 0:
        classes = obb.cls.cpu().numpy().astype(int).tolist()
        confidences = obb.conf.cpu().numpy().tolist()
        for class_id, confidence in zip(classes, confidences):
            detections.append(
                {
                    "label": names.get(class_id, str(class_id)),
                    "confidence": float(confidence),
                }
            )

    if not detections:
        return {
            "predicted_label": "No cavity detected",
            "predicted_confidence": 0.0,
            "report": "The model did not detect a cavity region above the confidence threshold. Review the image clinically before making any decision.",
            "top_predictions": [],
        }

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    counts = Counter(item["label"] for item in detections)
    top_detection = detections[0]
    cavity_count = counts.get("cavity", 0)
    normal_count = counts.get("normal", 0)

    if cavity_count > 0:
        summary = (
            f"The model detected {cavity_count} cavity candidate region"
            f"{'s' if cavity_count != 1 else ''}. The highest-confidence finding is "
            f"{top_detection['label']} at {top_detection['confidence'] * 100:.2f}%. "
            "Use this as AI assistance only; a dentist should confirm the finding."
        )
    else:
        summary = (
            f"The model detected {normal_count} normal tooth region"
            f"{'s' if normal_count != 1 else ''} and no cavity candidate above the threshold. "
            "Use this as AI assistance only; a dentist should confirm the image."
        )

    return {
        "predicted_label": top_detection["label"],
        "predicted_confidence": top_detection["confidence"],
        "report": summary,
        "top_predictions": detections[:8],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    report = None
    image_name = None

    if request.method == "POST":
        uploaded_file = request.files.get("xray_image")
        if uploaded_file is None or uploaded_file.filename == "":
            error = "Select an X-ray image to continue."
        elif not allowed_file(uploaded_file.filename):
            error = "Unsupported file type. Use PNG, JPG, JPEG, BMP, TIF, or TIFF."
        else:
            filename = secure_filename(uploaded_file.filename)
            image_path = UPLOAD_DIR / filename
            uploaded_file.save(image_path)
            image_name = filename

            try:
                report = build_detection_report(image_path)
            except Exception as exc:
                error = str(exc)

    model_path = resolve_model_path() if MODEL_PATH.exists() or FALLBACK_MODEL_PATH.exists() else MODEL_PATH
    status = {
        "model_exists": model_path.exists(),
        "labels_exist": True,
        "model_path": str(model_path),
        "labels_path": "Labels are embedded in the YOLO model.",
    }

    return render_template(
        "index.html",
        error=error,
        report=report,
        image_name=image_name,
        status=status,
    )


@app.route("/health")
def health():
    model_path = resolve_model_path() if MODEL_PATH.exists() or FALLBACK_MODEL_PATH.exists() else MODEL_PATH
    return {
        "status": "ok",
        "model_exists": model_path.exists(),
        "model_path": str(model_path),
        "uploads_dir": str(UPLOAD_DIR),
    }


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
