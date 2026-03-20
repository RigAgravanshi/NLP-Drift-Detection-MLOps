import yaml
import logging
from src.utils.logger import get_logger

def check_drift(drift_score, threshold, column_name):
    if drift_score > threshold :
        logger = get_logger(__name__)
        logger.warning(f"Drift detected on '{column_name}' column. Score: {drift_score:.4f}, Threshold: {threshold}")
        return True
    else: return False

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    text_drift_score = 0.60
    label_drift_score = 0.10      
    # these scores are only for simulating the warning and threshold working
    # real scores have been used and implemented in drift_detector.py
    result_text = check_drift(text_drift_score, config["monitoring"]["text_drift_threshold"], "text") 
    result_label = check_drift(label_drift_score, config["monitoring"]["label_drift_threshold"], "label")
    print(f"Drift detected: Text={result_text}, Label={result_label}")

if __name__ == "__main__":
    main()