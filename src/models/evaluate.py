import yaml
import pandas as pd
import src.utils.mlflow_utils as mlu
from src.utils.logger import get_logger
from src.models.predict import BertInference
from sklearn.metrics import accuracy_score, f1_score

logger = get_logger(__name__)
def evaluate(model, X_test, y_test):

    confidence = []
    intent_preds = []  
    intent_true = y_test.to_list()          
    for i in range(len(X_test)):
        preds, conf = model.predict(X_test[i])
        intent_preds.append(preds)
        confidence.append(conf)
    
    intent_acc = accuracy_score(intent_true, intent_preds)
    intent_f1 = f1_score(intent_true, intent_preds, average="weighted")
    mlu.log_metrics({"accuracy": intent_acc, "f1 score":intent_f1})
    logger.info(f"Accuracy: {intent_acc:.4f} \nf1 score: {intent_f1:.4f}")
    
def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = BertInference(config['model']['classifier_name'], config['model']['num_classes'])
    test_df = pd.read_csv(config["paths"]["production_data_path"])
    X_test = test_df["text"]
    y_test = test_df["label"]

    with mlu.start_run("model-evaluation", "run-1"):
        evaluate(model, X_test, y_test)
        logger.info("Evaluation Completed!")

if __name__ == "__main__":
    main()