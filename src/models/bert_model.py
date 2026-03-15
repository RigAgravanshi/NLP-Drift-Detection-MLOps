import torch.nn as nn
from transformers import AutoModel

class BertClassifier(nn.Module):
    # This class recieves a classifier name, and intent classes number and returns a classifier model
    def __init__(self, classifier_name, num_intent_classes):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(classifier_name)
        hidden_state = self.bert.config.hidden_size
        self.intent_classifier = nn.Linear(hidden_state, num_intent_classes)   
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask  
        )                                                                       
        cls_embeddings = outputs.last_hidden_state[:, 0, :]                   
        intent_logits =  self.intent_classifier(cls_embeddings)
        return intent_logits