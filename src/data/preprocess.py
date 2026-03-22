import yaml
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)
def preprocess(df):
    clean_df = df.copy()
    clean_df = clean_df.drop_duplicates(ignore_index = True)
    clean_df = clean_df.dropna()
    clean_df["text"] = clean_df["text"].str.strip()
    logger.info("Successful Data preprocessing")
    return clean_df

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_df = pd.read_csv(config["paths"]["reference_data_path"])
    clean_df = preprocess(train_df)
    clean_df.to_csv(config["paths"]["reference_data_path"], index = False)
    
if __name__ == "__main__":
    main()