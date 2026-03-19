import yaml
import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from src.monitoring.thresholds import check_drift

def run_drift_report(config, reference_df, production_df, report_name):
    data_definition = DataDefinition(
        categorical_columns=['label'],
        text_columns=['text']
    )
    reference = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current = Dataset.from_pandas(production_df, data_definition=data_definition)
    
    report = Report([DataDriftPreset()])
    result = report.run(reference, current)
    result.save_html(f"reports/{report_name}.html")
    result.save_json(f"reports/{report_name}.json")

    result_dict = result.dict()
    for metric in result_dict["metrics"]:
        if metric["config"].get("column") == "text":
            text_drift_score = metric["value"]
        if metric["config"].get("column") == "label":
            label_drift_score = metric["value"]
    check_drift(text_drift_score, config["monitoring"]["text_drift_threshold"], "text") 
    check_drift(label_drift_score, config["monitoring"]["label_drift_threshold"], "label")

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    reference_df = pd.read_csv(config['paths']['reference_data_path'])

    data_drift = pd.read_parquet(config['paths']["production_data_data_path"])
    label_drift = pd.read_parquet(config['paths']["production_data_label_path"])
    concept_drift = pd.read_parquet(config['paths']["production_data_concept_path"])

    run_drift_report(config, reference_df, data_drift, "data_drift_report")
    run_drift_report(config, reference_df, label_drift, "label_drift_report")
    run_drift_report(config, reference_df, concept_drift, "concept_drift_report")


if __name__ == "__main__":
    main()