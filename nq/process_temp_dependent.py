import re
import os
import tqdm

def filter_questions(all_questions):
    preps = {'on', 'in', 'from', 'until'}
    filtered_questions = []
    diffs = []
    for question in all_questions:
        q_toks = [(tok, tok in preps, bool(re.match("\d{4}($|\s|s|'s)", tok)))
                  for tok in question.split()]
        f_toks = []
        diff = False
        super_diff = False
        for i, tok in enumerate(q_toks):
            if tok[1]:
                idx = i+1
                mul = False
                if i+1 < len(q_toks) and q_toks[i+1][2]:
                    diff = True
                    continue
            if tok[2]:
                diff = True
                continue
            f_toks.append(tok[0])
        filtered_q =  ' '.join(f_toks)
        filtered_questions.append(filtered_q)
        if diff:
            diffs.append(question)
            diffs.append(filtered_q)
    return filtered_questions, diffs

def filter_questions_basic(all_questions):
    preps = {'on', 'in', 'from', 'until'}
    filtered_questions = []
    diffs = []
    for question in all_questions:
        filtered_q = re.sub('(in)?\s\d{4}$', '', question).strip()
        filtered_questions.append(filtered_q)
        if filtered_q != question:
            diffs.append(question)
            diffs.append(filtered_q)
    return filtered_questions, diffs


sent_dir = f'/data/mjqzhang/question_generation'
split = 'dev'
split = 'train'
with open(os.path.join(sent_dir, f'nq_filtered.{split}.txt'), 'r') as sent_tgt_f:
    all_questions = [q.strip() for q in sent_tgt_f]

# filtered_questions, diffs = filter_questions(all_questions)
filtered_questions, diffs = filter_questions_basic(all_questions)

with open(os.path.join(sent_dir, f'nq_filtered.{split}.fil'), 'w') as out_f:
    for question in filtered_questions:
        out_f.write(question + '\n')

with open(os.path.join(sent_dir, f'nq_filtered.{split}.dif'), 'w') as out_f:
    for question in diffs:
        out_f.write(question + '\n')
