from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from nlp import load_dataset
import torch
import numpy as np


model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

import random

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length = 256, add_special_tokens=True, padding='max_length', return_attention_mask=True)

train_dataset = load_dataset('json', data_files={'train':'dataset_full_question/quanta_train.json'}, field='questions')['train']
train_dataset = train_dataset.map(lambda example: {'label': [0 if example['difficulty'] == 'School' else 1]})
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset) )
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
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
    train_dataset=train_dataset,
)

trainer.train()
model.save_pretrained('models/DistilBERT_full_question')
