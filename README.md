# NLP Drift Detection — MLOps Pipeline

> Most ML projects stop at model training. This one doesn't. ## change this shitty line

This project fine-tunes BERT on the Banking77 dataset for 77-class intent 
classification, then builds a complete MLOps pipeline around it — serving 
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
                              │          Drift Detection Pipeline        │
                              │                                          │
                              │  Reference Data ──▶ Evidently AI ◀── Production Data  │
                              │                          │               │
                              │                    Drift Report          │
                              │                    + Threshold Check     │
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
│   │   ├── predict.py           # BertInference — model serving class
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
├── reports/                     # Sample Evidently drift reports
├── Dockerfile
├── docker-compose.yml
├── Makefile                     # Shortcuts for all commands            
├── requirements.txt
└── README.md


```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
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

**Key finding:** Concept drift cannot be detected by comparing input distributions 
alone — model predictions are required. This is a fundamental limitation of 
input-only monitoring and a known challenge in production ML systems.

### Sample Reports
See `reports/examples/` for full HTML drift reports.

---

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
- [ ] Real inference logging — log every API prediction to parquet
- [ ] Automated retraining trigger when drift exceeds threshold
- [ ] Cloud deployment — AWS EC2 or Railway
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Streamlit dashboard for drift monitoring UI
- [ ] DVC for data versioning

---

Built as an MLOps portfolio project demonstrating production ML engineering 
practices — model serving, project life-cycle awareness, drift detection, experiment tracking, and 
containerization.