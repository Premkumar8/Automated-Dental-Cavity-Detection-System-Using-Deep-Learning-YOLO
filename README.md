# Dental X-Ray Disease Classifier

This project trains a TensorFlow model on the Kaggle dataset:

`lokisilvres/dental-disease-panoramic-detection-dataset`

The output for a single panoramic X-ray is:

- predicted dental disease label
- confidence scores
- a short AI-generated report based on the image

## Expected classes

The code is dataset-driven and reads class names from folders automatically. A fallback disease knowledge base is included for these seven labels commonly associated with this dataset:

- `caries`
- `deep caries`
- `impacted tooth`
- `periapical lesion`
- `periodontal disease`
- `root resorption`
- `missing teeth`

## Files

- `download_dataset.py`: download the Kaggle dataset with retry support
- `train_model.py`: train and evaluate the classifier
- `predict_report.py`: run inference on one X-ray and generate a report
- `app.py`: local web app for uploading an X-ray and viewing the report
- `dental_xray_utils.py`: shared dataset, model, and report logic

## Setup

Use the existing Python 3.10 install:

```powershell
& 'C:\Users\karthiksuresh\AppData\Local\Programs\Python\Python310\python.exe' -m pip install -r requirements.txt
```

## Download dataset

```powershell
& 'C:\Users\karthiksuresh\AppData\Local\Programs\Python\Python310\python.exe' download_dataset.py
```

If Kaggle download fails mid-transfer, rerun the command. The script reports the cache path so you can inspect the downloaded files.

## Train

```powershell
& 'C:\Users\karthiksuresh\AppData\Local\Programs\Python\Python310\python.exe' train_model.py --dataset-dir "<dataset_root>" --epochs 15
```

Example:

```powershell
& 'C:\Users\karthiksuresh\AppData\Local\Programs\Python\Python310\python.exe' train_model.py --dataset-dir "C:\Users\karthiksuresh\.cache\kagglehub\datasets\lokisilvres\dental-disease-panoramic-detection-dataset"
```

Artifacts are written to `artifacts/`:

- `best_model.keras`
- `final_model.keras`
- `class_names.json`
- `metrics.json`
- `history.json`

## Predict and generate report

```powershell
& 'C:\Users\karthiksuresh\AppData\Local\Programs\Python\Python310\python.exe' predict_report.py --image "<xray_path>" --model artifacts\best_model.keras --labels artifacts\class_names.json
```

This writes:

- console summary
- `artifacts/latest_report.json`

## Run the web page

```powershell
& 'C:\Users\karthiksuresh\AppData\Local\Programs\Python\Python310\python.exe' app.py
```

Then open:

`http://127.0.0.1:5000`

The web UI expects these files from training:

- `artifacts/best_model.keras`
- `artifacts/class_names.json`

## Notes

- This is an AI assistance model, not a clinical diagnosis system.
- Real performance depends on full dataset download, class balance, and image quality.
- For production use, a dentist or oral radiologist must validate predictions.
