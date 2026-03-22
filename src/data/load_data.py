from datasets import load_dataset
from src.utils.logger import get_logger
import yaml

logger = get_logger(__name__)

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset = load_dataset("banking77")
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()

    train_df.to_csv(config["paths"]["reference_data_path"], index=False)
    test_df.to_csv(config["paths"]["production_data_path"], index= False)
    logger.info("Successful loading of Train and Test data")
    
if __name__ == "__main__":
    main()