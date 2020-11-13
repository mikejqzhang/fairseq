import os
import re
import csv
import json
import spacy
from tqdm import tqdm

MAX_SA_TOKENS = 5

CONTEXT_TOK = '[CTX]'
ANSWER_TOK = '[ANS]'

def get_questions(data):
    # nlp = spacy.load("en_core_web_sm")
    question_data = []
    
    n_tables_with_dates = 0
    for i_ex, ex in enumerate(tqdm(data)):
        table_has_date = \
            any(any(bool(re.search('\d{4}', cell['value'])) for cell in row) for row in ex['table'])
        if not table_has_date:
            continue

        highlighted_values = \
                [ex['table'][r][c]['value'] for r, c in ex['highlighted_cells']]
        filtered_values = \
                [v for v in highlighted_values if not re.search('\d{4}', v)]
        title = ex['table_page_title']
        summary = ex['sentence_annotations'][0]['final_sentence']

        filtered_summary = re.sub('in \d{4}', '', summary)
        filtered_summary = re.sub('\d{4}', '', filtered_summary)
        for v in highlighted_values:
            filtered_summary = re.sub(re.escape(v), '', filtered_summary)

        for ans in highlighted_values:
            sent_ex = {'context': summary.strip(),
                       'filtered_context': filtered_summary.strip(),
                       'title': title,
                       'answer': ans,
                       'highlighted_values': highlighted_values,
                       }
            question_data.append(sent_ex)
    return question_data

input_path = '/data/mjqzhang/totto_data/totto_{}_data.jsonl'
output_dir = '/data/mjqzhang/question_generation/totto_qgen'
split = 'dev'
# split = 'train'

with open(input_path.format(split), 'r') as f:
    data = [json.loads(l) for l in f]

question_data = get_questions(data)

# os.makedirs(output_dir, exist_ok=True)
# with open(os.path.join(output_dir, f'{split}.src'), 'w') as output_src_f:
#     for ex in question_data:
#         title = ex['title']
#         context = ex['context']
#         answer = ex['answer']
#         output_src_f.write(f'{title} {CONTEXT_TOK} {context} {ANSWER_TOK} {answer}\n')
# 
# with open(os.path.join(output_dir, f'{split}.hvs'), 'w') as output_src_f:
#     for ex in question_data:
#         output_src_f.write(json.dumps(ex['highlighted_values']) + '\n')
# 
# with open(os.path.join(output_dir, f'{split}.fsrc'), 'w') as output_src_f:
#     for ex in question_data:
#         title = ex['title']
#         context = ex['filtered_context']
#         answer = ex['answer']
#         output_src_f.write(f'{title} {CONTEXT_TOK} {context} {ANSWER_TOK} {answer}\n')
