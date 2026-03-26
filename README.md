# NLP Drift Detection — MLOps Pipeline

> What happens post-deployment of ML/DL models and applications? This project is an exploration of beyond the journey's end

This project fine-tunes BERT on the Banking77 dataset for 77-class intent 
classification, then builds a complete MLOps pipeline around it; serving 
predictions via a FastAPI REST API, simulating three types of production drift 
(data, label, and concept), detecting drift using Evidently AI, and tracking 
all metrics through MLflow. The entire pipeline is containerized with Docker.

---

## Architecture
```
Raw Text Input
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  FastAPI    │────▶│ BertClassifier│────▶│ Intent + Score  │
│  /predict   │     │  (inference)  │     │  (77 classes)   │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                  │
                                    ┌─────────────▼──────────────┐
                                    │     Inference Logs          │
                                    │  data/inference_logs/       │
                                    └─────────────┬──────────────┘
                                                  │
                              ┌───────────────────▼────────────────────┐
                              │          Drift Detection Pipeline      │
                              │                                        │
                       │ Reference Data ──▶ Evidently AI ◀── Production Data │
                              │                   │                    │
                              │                Drift Report            │
                              │            + Threshold Check           │
                              └───────────────────┬────────────────────┘
                                                  │
                                    ┌─────────────▼──────────────┐
                                    │        MLflow Tracking      │
                                    │  • Training metrics         │
                                    │  • Drift scores             │
                                    │  • Parameters               │
                                    └────────────────────────────┘
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch + HuggingFace | BERT fine-tuning and inference |
| FastAPI | REST API for model serving |
| Evidently AI | Drift detection and reporting |
| MLflow | Experiment and drift metric tracking |
| Docker | Containerization |
| Pydantic | Request/response validation |

---

## Project Structure
```
nlp-mlops-pipeline/
├── configs/config.yaml          # All settings, no hardcoded values
├── notebooks                    # experimental Jupyter training notebook
├── src/
│   ├── models/
│   │   ├── bert_model.py        # BertClassifier architecture
│   │   ├── predict.py           # BertInference model serving class
│   │   ├── train.py             # Training pipeline with MLflow
│   │   └── evaluate.py          # Evaluation pipeline
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── monitoring/
│   │   ├── simulate_drift.py    # Generates 3 drift scenarios
│   │   ├── drift_detector.py    # Evidently drift reports
│   │   └── thresholds.py        # Alert logic
│   ├── data/
│   │   ├── load_data.py         # Pull Banking77 from HuggingFace
│   │   └── preprocess.py        # Clean and validate data
│   └── utils/
│       ├── logger.py            # Structured logging to file + terminal
│       └── mlflow_utils.py      # MLflow helper functions
├── data/
│   ├── reference/               # Training split — drift baseline
│   ├── production/              # Simulated drift batches
│   └── inference_logs/          # Logged API predictions
├── tests/
│   ├── test_api.py               # tests the running of api
│   ├── test_drift.py             # tests the drift detection pipeline(loads mock model)
│   └── test_preprocessing        # test the preprocessing pipeline
├── scripts/                      # shell files in bash script
├── reports/                      # Sample Evidently drift reports
├── Dockerfile
├── docker-compose.yml
├── Makefile                      # Shortcuts for all commands            
├── requirements.txt
└── README.md


```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the repo
```bash
git clone https://github.com/RigAgravanshi/NLP-Drift-Detection-MLOps.git
cd NLP-Drift-Detection-MLOps
```

### 2. Install dependencies
```bash
make setup
```

### 3. Pull data
```bash
make pull-data
```

### 4. Preprocess
```bash
make preprocess
```

### 5. Simulate drift scenarios
```bash
make simulate
```

---

## Usage

### Run the API
```bash
make serve
```
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

**Example request:**
```json
POST /predict
{
  "text": "I lost my card and need a replacement immediately"
}
```

**Example response:**
```json
{
  "predicted_intent": "lost_or_stolen_card",
  "confidence": 94.32
}
```

### Run drift detection
```bash
make drift
```

### Launch MLflow UI
```bash
make mlflow-ui
```
Visit `http://127.0.0.1:5000`

### Train model
```bash
make train
```

### Evaluate model
```bash
make evaluate
```

---

## Drift Detection Results

Three drift scenarios are simulated from the Banking77 test split:

| Scenario | Method | Text Drift Score | Label Drift Score | Detected |
|----------|--------|-----------------|-------------------|----------|
| Data Drift | Typos + abbreviations on 40% of texts | 0.634 | 0.096 | ✅ Text |
| Label Drift | Oversample 4 intents to 70% of data | 0.601 | 0.221 | ✅ Both |
| Concept Drift | Shuffle labels on 40% of samples | 0.519 | 0.096 | ❌ None |
<img width="1264" height="858" alt="image" src="https://github.com/user-attachments/assets/73bce23d-deeb-45f0-8c51-d6cc239dd902" />
MLFlow tracking experiment


### Sample Reports

**Key findings:**

Data drift report — 1 column drifted (text):
<img width="1770" height="489" alt="image" src="https://github.com/user-attachments/assets/11180e82-7482-4900-bbb7-38728ec7e3b3" />


We only corrupted the text by adding typos and abbreviations. Labels were untouched. So only the text column shows drift.

Label drift report — 2 columns drifted (text + label):
<img width="1774" height="480" alt="image" src="https://github.com/user-attachments/assets/11f803b6-e942-4d57-ad3f-57b20bbc36ba" />


We oversampled 4 intents heavily. That changed the label distribution and so label drift is detected. But oversampling also means those same texts appear 6 times each in production data. The text distribution shifts too because the same sentences are now repeated far more than in reference. So text drift is a side effect of oversampling. Both columns flag correctly.

Concept drift report — 0 columns drifted:
<img width="1776" height="489" alt="image" src="https://github.com/user-attachments/assets/cd15a48c-7e3a-4b0e-8867-32bb190b3fff" />


We only shuffled labels on 40% of rows. Texts are identical to reference. Evidently sees no input change. Concept drift is invisible to input-only monitoring. Concept drift cannot be detected by comparing input distributions 
alone, model predictions are required.

**Note**: Thresholds were set empirically based on observed baseline scores. In production I would establish thresholds statistically by measuring drift variance across reference data splits and setting the threshold at the 95th percentile.


## MLflow Tracking

All training runs and drift detection runs are tracked in MLflow.

**Tracked parameters:** model name, epochs, batch size, learning rate, 
drift thresholds

**Tracked metrics:** training loss per epoch, accuracy, F1 score, 
text drift score, label drift score per scenario
```bash
make mlflow-ui
```

---

## Running with Docker
```bash
docker-compose up
```

- API: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`

---

## Future Improvements

- [ ] Embedding-based drift detection using BERT CLS vectors + PCA
- [ ] Real inference logging —> log every API prediction to parquet
- [ ] Automated retraining trigger when drift exceeds threshold
- [ ] Cloud deployment —> AWS EC2
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Streamlit dashboard for drift monitoring UI

---

Built as an MLOps project demonstrating production ML engineering 
practices like model serving, project life-cycle awareness, drift detection, experiment tracking, and 
containerization.
