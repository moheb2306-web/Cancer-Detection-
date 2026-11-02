# Brain Tumor Detection (CNN + FastAPI)

Binary MRI classifier (tumor / no tumor). `train.py` trains a custom CNN and a
MobileNetV2 transfer-learning model, picks the best by **validation AUC**,
reports on a held-out **test set**, and saves it. `app/main.py` serves it.

## Layout
```
brain-tumor-detection/
├── data/yes/*.jpg      # tumor images
├── data/no/*.jpg       # no-tumor images
├── train.py
├── app/main.py
├── models/             # created by train.py
├── requirements.txt
└── Dockerfile
```

## 1. Train (Google Colab)
```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/brain-tumor-detection
!pip install -q -r requirements.txt
!python train.py --data_dir data --epochs 30
```
Outputs in `models/`: `best_model.keras`, `model_meta.json`,
`training_curves.png`, `confusion_matrix.png`. Download `models/` to run the API locally.

## 2. Serve
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs, or:
```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@scan.jpg"
```
Response:
```json
{"label":"tumor","tumor_probability":0.9312,"confidence":0.9312,
 "threshold":0.5,"model_name":"mobilenetv2","disclaimer":"..."}
```

## Notes
- With ~250 images, results vary a lot between splits; trust the test-set
  report and confusion matrix over accuracy alone. Recall on "tumor" matters most
  here: lower `--threshold` (e.g. 0.35) to trade precision for fewer missed tumors.
- The `brain_tumor_synthetic_data.csv` is tabular and is not used by the image CNN.
- Research/educational use only, not a medical device.
