import yaml
import torch
import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import src.utils.mlflow_utils as mlu
from src.utils.logger import get_logger
from src.models.bert_model import BertClassifier

logger = get_logger(__name__)
def train(model, device, epochs, optimizer, intent_loss_fn, train_loader):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent_labels"].to(device)
    
            intent_logits = model(input_ids, attention_mask)

            intent_loss = intent_loss_fn(intent_logits, intent_labels)

            loss = intent_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"\nEpoch: {epoch+1}/{epochs} \nTraining loss: {avg_loss}")
        mlu.log_metrics({"epoch": epoch+1, "training_loss": avg_loss})

class TicketDataset(Dataset):
    def __init__(self, encodings, intent_labels):
        self.encodings = encodings
        self.intent_labels = intent_labels

    def __len__(self):
        return len(self.intent_labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "intent_labels": torch.tensor(self.intent_labels[idx])
        }
        return item

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # defining the parameters required for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertClassifier(config["model"]["classifier_name"], config["model"]["num_classes"]).to(device)
    epochs = config["training"]["epochs"]
    optimizer = AdamW(model.parameters(), lr=config["training"]["lr"])
    intent_loss_fn = nn.CrossEntropyLoss()

    # loading train_df and splitting into X(text data) and y(target labels)
    train_df = pd.read_csv(config["paths"]["reference_data_path"])
    X_train = train_df["text"]
    y_train = train_df["label"]

    # Creating the tokenizer based Dataset and using DataLoader
    tokenizer = AutoTokenizer.from_pretrained(config['paths']['tokenizer_path'])
    train_encodings = tokenizer(X_train.to_list(), truncation=True, padding=True, max_length=config["model"]["max_length"])

    train_dataset = TicketDataset(train_encodings, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    #MLFlow function to define the exp and run name
    with mlu.start_run("bert-intent-classifier", "run-1"):
        mlu.log_params({"model":config["model"]["classifier_name"], "epochs":epochs,
            "learning_rate":config["training"]["lr"],
            "max_length":config["model"]["max_length"],
            "batch_size":config["training"]["batch_size"]
            })
        
        # function call and weights saving
        train(model, device, epochs, optimizer, intent_loss_fn, train_loader)
        torch.save(model.state_dict(), config["paths"]["model_path"])
        logger.info("Model Trained and Saved Successfully!")


if __name__ == "__main__":
    main()