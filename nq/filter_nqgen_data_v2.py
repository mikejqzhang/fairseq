import re
import os
import json
import random
from tqdm import tqdm

CONTEXT_TOK = '[CTX]'
ANSWER_TOK = '[ANS]'
RANDOM_SEED = 88888888

random.seed(RANDOM_SEED)
sent_dir = f'/data/mjqzhang/question_generation/nqgen_sent_v2'
# split = 'train'
split = 'dev'

with open(os.path.join(sent_dir, f'{split}.jsonl'), 'r') as f:
    all_data = [json.loads(l) for l in f]
random.shuffle(all_data)

new_questions = []
diffs = []
preps = ' |'.join(['on', 'in', 'from', 'until', 'for', 'after', 'before']) + ' '
remove_date_pattern = "(^| )(" + preps + ")?\d{4}( |$)"
remove_date_s_pattern = "(^| )(" + preps + ")?(the )?(early |late |mid )?\d{4}(s|'s)"

for ex in tqdm(all_data):
    question = ex['question']
    # rewrite = None
    # if re.search(has_date_pattern, question):
    #     rewrite = re.sub(remove_date_pattern, ' ', question)
    #     new_questions.append(rewrite)
    #     diffs.append(question)
    #     diffs.append(rewrite)
    # else:
    #     new_questions.append(question)
    rewrite = re.sub(remove_date_s_pattern, '', question)
    rewrite = re.sub(remove_date_pattern, '', rewrite).strip()
    if rewrite != question:
        diffs.append(question)
        diffs.append(rewrite)
    new_questions.append(rewrite)
    ex['question_rewrite'] = rewrite

with open(os.path.join(sent_dir, f'{split}.new'), 'w') as out_f:
    for question in new_questions:
        out_f.write(question + '\n')

with open(os.path.join(sent_dir, f'{split}.dif'), 'w') as out_f:
    for question in diffs:
        out_f.write(question + '\n')

with open(os.path.join(sent_dir, f'{split}.src'), 'w') as src_f:
    for ex in all_data:
        src_line = f"{ex['title']} {CONTEXT_TOK} {ex['context']} {ANSWER_TOK} {ex['answer']}\n"
        src_f.write(src_line)

with open(os.path.join(sent_dir, f'{split}.tgt'), 'w') as tgt_f:
    for ex in all_data:
        tgt_line = f"{ex['question_rewrite']}\n"
        tgt_f.write(tgt_line)
