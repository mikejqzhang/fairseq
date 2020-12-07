import json
from tqdm import tqdm
from collections import Counter

import torch
from transformers import BertTokenizer, BertForSequenceClassification

dev_path = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_ans_in_context.out.jsonl'
test_path = '/data/mjqzhang/mturk_data/good_responses.jsonl'

model_path = '/data/mjqzhang/question_generation/totto_qgen/classifier/outputs/checkpoint-{}/'

class ToTToDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = [json.loads(l) for l in f]
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        texts = [ex['question'] for ex in self.data]
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = [int(ex['label']) for ex in self.data]
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


dev_dataset = ToTToDataset(dev_path)
test_dataset = ToTToDataset(test_path)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=700, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=700, shuffle=False)

device = torch.device('cuda')

model = BertForSequenceClassification.from_pretrained(model_path.format(6000))
model.to(device)
model.eval()
dev_preds = []
dev_labels = []
for batch in dev_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    preds = outputs[1].argmax(dim=-1)
    dev_preds.extend(preds.tolist())
    dev_labels.extend(labels.tolist())

test_preds = []
test_labels = []
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    preds = outputs[1].argmax(dim=-1)
    test_preds.extend(preds.tolist())
    test_labels.extend(labels.tolist())
print('dev_confusion')
print({k: v / len(dev_labels) for k, v in Counter(zip(dev_preds, dev_labels)).items()})
print('test_confusion')
print({k: v / len(test_labels) for k, v in Counter(zip(test_preds, test_labels)).items()})


# with open('results_log.jsonl', 'w') as f:
#     for i in tqdm(range(500, 24000, 500)):
#         model = BertForSequenceClassification.from_pretrained(model_path.format(i))
#         model.to(device)
#         model.eval()
#         dev_correct = 0
#         dev_total = 0
#         for batch in dev_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             preds = outputs[1].argmax(dim=-1)
#             dev_correct += (preds == labels).sum().item()
#             dev_total += len(labels)
# 
#         test_correct = 0
#         test_total = 0
#         for batch in test_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             preds = outputs[1].argmax(dim=-1)
#             test_correct += (preds == labels).sum().item()
#             test_total += len(labels)
# 
#         results = {'checkpoint': i,
#                    'dev_acc': dev_correct / dev_total,
#                    'test_acc': test_correct / test_total}
#         print(json.dumps(results, indent=2))
#         f.write(json.dumps(results) + '\n')

