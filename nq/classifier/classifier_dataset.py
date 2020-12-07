import json

import torch
from transformers import BertTokenizer

def preprocess_totto(tokenizer, path, out_path):
    with open(path, 'r') as f:
        data = [json.loads(l) for l in f]
    texts = [ex['question'] for ex in data]
    tokens = tokenizer(texts, truncation=True, padding=True)
    labels = [int(ex['label']) for ex in data]
    dataset = {'encodings': tokens, 'labels': labels}
    with open(out_path, 'w') as f:
        json.dump(dataset, f)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_path = '/data/mjqzhang/question_generation/totto_qgen/outputs/train_ans_in_context.out.jsonl'
    dev_path = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_ans_in_context.out.jsonl'

    train_out_path = '/data/mjqzhang/question_generation/totto_qgen/classifier/train_ans_in_context.jsonl'
    dev_out_path = '/data/mjqzhang/question_generation/totto_qgen/classifier/dev_ans_in_context.jsonl'

    preprocess_totto(tokenizer, dev_path, dev_out_path)
    preprocess_totto(tokenizer, train_path, train_out_path)
