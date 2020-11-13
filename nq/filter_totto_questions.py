import csv
import json

src_path = '/data/mjqzhang/question_generation/totto_qgen/v2/dev_maxans_3.jsonl'
hyp_path = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_maxans_3.hyp'
out_path = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_maxans_3.csv'
# hyp_path = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_og_maxans_3.hyp'
# out_path = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_og_maxans_3.csv'
# hyp_path = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_del_maxans_3.hyp'
# out_path = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_del_maxans_3.csv'
n_questions = 2500

with open(src_path, 'r') as f:
    src_data = []
    for i in range(n_questions):
        src_data.append(json.loads(f.readline()))
with open(hyp_path, 'r') as f:
    hyp_data = []
    for i in range(n_questions):
        hyp_data.append(json.loads(f.readline()))

output_rows = []
for srcl, hypl in zip(src_data, hyp_data):
    hyp_has_unique_tok = [x for x in hypl[0].split() if x in  srcl['unique_tokens']]
    row = [
            hypl[0],
            hyp_has_unique_tok,
            srcl['context'],
            # srcl['og_context'],
            # srcl['del_context'],
            srcl['title'],
            srcl['answer'],
            srcl['table_has_date'],
            srcl['n_highlighted_values'],
            ]
    output_rows.append(row)

with open(out_path, 'w') as f:
    writer = csv.writer(f)
    for row in output_rows:
        writer.writerow(row)
