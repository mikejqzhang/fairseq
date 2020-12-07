import re
import os
import json
import random
from tqdm import tqdm

CONTEXT_TOK = '[CTX]'
ANSWER_TOK = '[ANS]'
RANDOM_SEED = 88888888
random.seed(RANDOM_SEED)

nqgen_dir = '/data/mjqzhang/question_generation/nqgen_extracted'
full_nqgen_dir = '/data/mjqzhang/question_generation/nqgen_full'
date_filtered_nqgen_dir = '/data/mjqzhang/question_generation/nqgen_date_filtered'

split = 'train'
# split = 'dev'

with open(os.path.join(nqgen_dir, f'{split}.jsonl'), 'r') as f:
    all_data = [json.loads(l) for l in f]
random.shuffle(all_data)

preps = ' |'.join(['on', 'in', 'from', 'until', 'for', 'after', 'before']) + ' '
remove_date_pattern = "(^| )(" + preps + ")?\d{4}( |$)"
remove_date_s_pattern = "(^| )(" + preps + ")?(the )?(early |late |mid )?\d{4}(s|'s)"

for ex in tqdm(all_data):
    question = ex['question']
    rewrite = re.sub(remove_date_s_pattern, '', question)
    rewrite = re.sub(remove_date_pattern, '', rewrite).strip()
    ex['question_date_filtered'] = rewrite

with open(os.path.join(nqgen_dir, f'{split}.processed_questions.jsonl'), 'w') as f:
    for ex in all_data:
        f.write(json.dumps(ex) + '\n')

with open(os.path.join(full_nqgen_dir, f'{split}.src'), 'w') as f:
    for ex in all_data:
        f.write(f"{ex['title']} {CONTEXT_TOK} {ex['context']} {ANSWER_TOK} {ex['answer']}\n")
with open(os.path.join(full_nqgen_dir, f'{split}.tgt'), 'w') as f:
    for ex in all_data:
        f.write(f"{ex['question']}\n")

with open(os.path.join(date_filtered_nqgen_dir, f'{split}.src'), 'w') as f:
    for ex in all_data:
        f.write(f"{ex['title']} {CONTEXT_TOK} {ex['context']} {ANSWER_TOK} {ex['answer']}\n")
with open(os.path.join(date_filtered_nqgen_dir, f'{split}.tgt'), 'w') as f:
    for ex in all_data:
        f.write(f"{ex['question_date_filtered']}\n")
