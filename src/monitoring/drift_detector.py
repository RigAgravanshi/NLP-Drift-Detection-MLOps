import yaml
import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

def run_drift_report(reference_df, production_df, report_name):
    data_definition = DataDefinition(
        categorical_columns=['label'],
        text_columns=['text']
    )
    reference = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current = Dataset.from_pandas(production_df, data_definition=data_definition)
    
    report = Report([DataDriftPreset()])
    result = report.run(reference, current)
    result.save_html(f"reports/{report_name}.html")


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    reference_df = pd.read_csv(config['paths']['reference_data_path'])

    data_drift = pd.read_parquet(config['paths']["production_data_data_path"])
    label_drift = pd.read_parquet(config['paths']["production_data_label_path"])
    concept_drift = pd.read_parquet(config['paths']["production_data_concept_path"])

    run_drift_report(reference_df, data_drift, "data_drift_report")
    run_drift_report(reference_df, label_drift, "label_drift_report")
    run_drift_report(reference_df, concept_drift, "concept_drift_report")


if __name__ == "__main__":
    main()