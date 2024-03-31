import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BertTextClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BertTextClassifier, self).__init__()
        self.bert_model = bert_model
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

    def predict_text(self, bert_tokenizer, sentence):
        # Tokenize the input sentence
        inputs = bert_tokenizer.encode_plus(sentence,
                                       max_length=130,
                                       add_special_tokens=True,
                                       return_token_type_ids=False,
                                       padding='max_length',
                                       return_attention_mask=True,
                                       return_tensors='pt', )
        # Run inference
        with torch.no_grad():
            logits = self(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            probs = torch.sigmoid(logits)
            prediction = (probs > 0.5).int().item()
        return prediction


def get_model():
    bert_tokenizer = BertTokenizer.from_pretrained('T4orty/idiom_classifier_bert')
    bert_model = BertModel.from_pretrained("T4orty/idiom_classifier_bert")
    classifier_model = BertTextClassifier(bert_model)
    return bert_tokenizer, classifier_model
