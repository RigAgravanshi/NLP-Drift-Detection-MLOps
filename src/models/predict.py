import torch
from src.models.bert_model import BertClassifier
from transformers import AutoTokenizer
import json
import yaml

class BertInference():
    def __init__(self, classifier_name, num_intent_classes):
        # super.init not needed bcoz we are not inheriting from any other class
        with open("configs/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_name = classifier_name
        self.num_intent_classes = num_intent_classes

        # load the tokenizer, model and label names
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['paths']['tokenizer_path'])
        self.model = BertClassifier(classifier_name, num_intent_classes).to(self.device)
        self.model.load_state_dict(torch.load(self.config['paths']['model_path'], map_location=self.device, weights_only=True))
        with open(self.config['paths']['label_names_path'], "r") as f:
            self.label_names = json.load(f)

    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt",
            truncation=True,
            padding=True,
            max_length = self.config['model']['max_length']
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

            confidence = probs[0][pred_class].item() * 100       # confidence of the prediction
            predicted_intent = self.label_names[str(pred_class)] # intent(converted to str; json o/p in label names)

            return predicted_intent, confidence