from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from nlp import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

model = BertForSequenceClassification.from_pretrained('models/BERT_last_line')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length = 128, add_special_tokens=True, padding='max_length', return_attention_mask=True)

test_dataset = load_dataset('json', data_files={'test': 'dataset_last_line/quanta_test.json'}, field='questions')['test']
test_dataset = test_dataset.map(lambda example: {'label': [0 if example['difficulty'] == 'School' else 1]})
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

def compute_metrics(pred):
    labels = pred.label_ids
    print(labels)
    preds = pred.predictions.argmax(-1)
    print(preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
    eval_dataset=test_dataset
)

print(trainer.evaluate())
