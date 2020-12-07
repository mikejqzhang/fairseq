import csv
import json
from tqdm import tqdm

split = 'train'
# split = 'dev'
CONTEXT_TOK = '[CTX]'
ANSWER_TOK = '[ANS]'

src_file = f'/data/mjqzhang/question_generation/totto_qgen/v2/{split}.jsonl'
hyp_file = f'/data/mjqzhang/question_generation/totto_qgen/outputs/{split}_ans_in_context.hyp'
out_file = f'/data/mjqzhang/question_generation/totto_qgen/outputs/{split}_ans_in_context.out.jsonl'
csv_file = f'/data/mjqzhang/question_generation/totto_qgen/outputs/{split}_ans_in_context.out.csv'

csv_rows = [['Generated Question', 'Source Title', 'Source Summary', 'Source Answer', 'Distant "is Temporal" Label']]
with open(src_file, 'r') as src_f, open(hyp_file, 'r') as hyp_f, open(out_file, 'w') as out_f:
    seen_questions = set()
    for src_line, hyp_line in zip(src_f, hyp_f):
        src_data, hyp_data = json.loads(src_line), json.loads(hyp_line)
        title, context, answer = src_data['title'], src_data['context'], src_data['answer']
        assert hyp_data['source'] == f'{title} {CONTEXT_TOK} {context} {ANSWER_TOK} {answer}'
        src_data.update(hyp_data)

        src_data['hypos_is_ambig'] = [not any(tok in hypo for tok in src_data['unique_tokens'])
                                      for hypo in src_data['hypos']]
        src_data['hypos_no_first'] = ['first' not in hypo for hypo in src_data['hypos']]
        src_data['hypos_labels'] = [(src_data['has_date'] and is_ambig and no_first)
                                    for is_ambig, no_first in
                                    zip(src_data['hypos_is_ambig'], src_data['hypos_no_first'])]
        src_data['question'] = src_data['hypos'][0]
        src_data['label'] = src_data['hypos_labels'][0]
        if src_data['question'] not in seen_questions:
            seen_questions.add(src_data['question'])
            out_f.write(json.dumps(src_data) + '\n')
            csv_rows.append([src_data['question'],
                             title,
                             context,
                             answer,
                             src_data['label'],
                             src_data['url'],
                             ])

with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)
