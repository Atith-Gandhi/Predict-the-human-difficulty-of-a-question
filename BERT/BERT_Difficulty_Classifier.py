from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments, DistilBertForSequenceClassification, DistilBertTokenizerFast
from nlp import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

import random

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length = 128, add_special_tokens=True, padding='max_length', return_attention_mask=True)

train_dataset_tmp, test_dataset_tmp = load_dataset('json', data_files={'train':'quanta_train.json', 'test': 'quanta_test.json'}, field='questions', split = ['train', 'test'])
train_dataset_tmp = train_dataset_tmp.shuffle(random.randint(0, 100))
test_dataset_tmp = test_dataset_tmp.shuffle(random.randint(0, 100))
train_dataset_tmp = train_dataset_tmp.map(lambda example: {'label': [True if example['difficulty'] == 'College' else False]})
test_dataset_tmp = test_dataset_tmp.map(lambda example: {'label': [True if example['difficulty'] == 'College' else False]})
train_dataset = train_dataset_tmp.map(tokenize, batched=True, batch_size=len(train_dataset_tmp) )
test_dataset = test_dataset_tmp.map(tokenize, batched=True, batch_size=len(test_dataset_tmp))
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
print(train_dataset)
print(test_dataset)

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

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    # warmup_steps = 0,
    # weight_decay=1e-8,
    learning_rate=1e-5,
    # evaluate_during_training=True,
    logging_dir='./logs',
    save_steps=500,
    logging_steps=500,
    do_eval=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

trainer.evaluate()
