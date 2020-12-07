import re
import os
import spacy
from tqdm import tqdm

sent_dir = f'/data/mjqzhang/question_generation'
split = 'dev'

with open(os.path.join(sent_dir, f'nq_filtered.{split}.txt'), 'r') as sent_tgt_f:
    all_questions = [q.strip() for q in sent_tgt_f]


filtered_questions = []
diffs = []
preps = {'on', 'in', 'from', 'until', 'the', 'early'}
possible_preps = set()

for question in tqdm(all_questions):
    q_toks = [(tok, tok in preps, bool(re.match("\d{4}($|\s|s|'s)", tok)))
              for tok in question.split()]
    f_toks = []
    diff = False
    super_diff = False
    for i, tok in enumerate(q_toks):
        if tok[1]:
            idx = i+1
            mul = False
            while idx < len(q_toks) and q_toks[idx][1]:
                idx += 1
                mul = True
            if idx < len(q_toks) and q_toks[idx][2]:
                diff = True
                if mul:
                    super_diff = True
                continue
            # if i+1 < len(q_toks) and q_toks[i+1][2]:
            #     diff = True
            #     continue
        if tok[2]:
            if i != 0:
                possible_preps.add(q_toks[i-1][0])
            diff = True
            continue
        f_toks.append(tok[0])
    filtered_q =  ' '.join(f_toks)
    filtered_questions.append(filtered_q)
    if diff:
        diffs.append(question)
        diffs.append(filtered_q)
        if super_diff:
            print(question)
            print(filtered_q)

print(possible_preps)

# with open(os.path.join(sent_dir, f'{split}.fil'), 'w') as out_f:
#     for question in filtered_questions:
#         out_f.write(question + '\n')
# 
# with open(os.path.join(sent_dir, f'{split}.diffs'), 'w') as out_f:
#     for question in diffs:
#         out_f.write(question + '\n')

with open(os.path.join(sent_dir, f'nq_filtered.{split}.fil'), 'w') as out_f:
    for question in filtered_questions:
        out_f.write(question + '\n')

with open(os.path.join(sent_dir, f'nq_filtered.{split}.dif'), 'w') as out_f:
    for question in diffs:
        out_f.write(question + '\n')
