import os
import json
import random

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

data_path = '/data/mjqzhang/mturk_data/good_responses.jsonl'
output_dir = '/data/mjqzhang/question_generation/totto_qgen/classifier/bert_baseline'

random.seed(88888888)
n_train, n_dev, n_test = 1000, 120, 120

class ToTToDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        print(len(data))
        self.data = data
        texts = [ex['question'] for ex in data]
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = [int(ex['label']) for ex in data]
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


with open(data_path, 'r') as f:
    data = [json.loads(l) for l in f]

random.shuffle(data)
train_data = data[:n_train]
dev_data = data[n_train:n_dev+n_train]
test_data = data[n_dev+n_train:]
assert len(test_data) == n_test

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print('Loading Training Dataset')
train_dataset = ToTToDataset(train_data, tokenizer)

print('Loading Dev Dataset')
dev_dataset = ToTToDataset(dev_data, tokenizer)

print('Loading Test Dataset')
test_dataset = ToTToDataset(test_data, tokenizer)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    do_eval=True,
    load_best_model_at_end=True
)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    )

print('Training...')
trainer.train()

print('Getting Dev Performance')
dev_predictions = trainer.predict(dev_dataset)
dev_pred_labels = dev_predictions[0].argmax(axis=1)
dev_gold_labels = dev_predictions[1]
dev_accuracy = (dev_pred_labels == dev_gold_labels).sum() / len(dev_gold_labels)
print('Accuracy:', dev_accuracy)

print('Getting Test Performance')
test_predictions = trainer.predict(test_dataset)
test_pred_labels = test_predictions[0].argmax(axis=1)
test_gold_labels = test_predictions[1]
test_accuracy = (test_pred_labels == test_gold_labels).sum() / len(test_gold_labels)
print('Accuracy:', test_accuracy)

with open(os.path.join(output_dir, 'dev_predicitons.jsonl'), 'w') as f:
    for x, l in zip(dev_data, dev_pred_labels.tolist()):
        x['baseline_pred_label'] = l
        f.write(json.dumps(x) + '\n')

with open(os.path.join(output_dir, 'test_predicitons.jsonl'), 'w') as f:
    for x, l in zip(test_data, test_pred_labels.tolist()):
        x['baseline_pred_label'] = l
        f.write(json.dumps(x) + '\n')

model.save_pretrained(os.path.join(output_dir, 'best_checkpoint'))
