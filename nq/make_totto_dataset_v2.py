import os
import re
import csv
import json
import spacy
from tqdm import tqdm


CONTEXT_TOK = '[CTX]'
ANSWER_TOK = '[ANS]'
TEMPORAL_HEADERS = ['year', 'date', 'time', 'season', 'term']
# DATE_REGEX = "(^| )\d{4}(s|'s| |$)"
DATE_REGEX = "(^|\W)\d{4}(\W|s|$)"
MAX_ANS = 3

def get_questions(data):
    question_data = []
    for i_ex, ex in enumerate(tqdm(data)):
        title = ex['table_page_title']
        title_has_date = bool(re.search(DATE_REGEX, title))

        table_headers = [
                cell['value'].lower() for row in ex['table'] for cell in row if cell['is_header']]
        header_has_date = any(header in TEMPORAL_HEADERS or re.search(DATE_REGEX, header)
                for header in table_headers)

        row_toks = [set(' '.join([cell['value'].lower()
            for cell in row if not cell['is_header']]).split()) for row in ex['table']]
        all_toks = set(tok for row in row_toks for tok in row)
        title_toks = set(title.lower().split())
        unique_tokens = [tok for tok in all_toks
                if sum(tok in row for row in row_toks) == 1 and tok not in title_toks]

        # row_has_date = [
        #         any(re.search(DATE_REGEX, cell['value']) for cell in row) for row in ex['table']]
        # row_has_date_perc = sum(row_has_date) / len(row_has_date)
        # table_has_date = any(row_has_date)
        table_has_date = any(
                bool(re.search(DATE_REGEX, cell['value'])) for row in ex['table'] for cell in row)

        highlighted_values = \
                [ex['table'][r][c]['value'] for r, c in ex['highlighted_cells']]

        if len(highlighted_values) <= MAX_ANS:
            for ans in highlighted_values:
                sent_ex = {'context': ex['sentence_annotations'][0]['final_sentence'],
                           'og_context': ex['sentence_annotations'][0]['original_sentence'],
                           'del_context': ex['sentence_annotations'][0]['sentence_after_deletion'],
                           'title': ex['table_page_title'],
                           'table_has_date': table_has_date,
                           'title_has_date': title_has_date,
                           'header_has_date': header_has_date,
                           'has_date': table_has_date or title_has_date or header_has_date,
                           'unique_tokens': unique_tokens,
                           # 'row_has_date_perc': row_has_date_perc,
                           'answer': ans,
                           'n_highlighted_values': len(highlighted_values),
                           'highlighted_values': highlighted_values}
                question_data.append(sent_ex)
    return question_data

input_path = '/data/mjqzhang/totto_data/totto_{}_data.jsonl'
output_dir = '/data/mjqzhang/question_generation/totto_qgen/v2'
os.makedirs(output_dir, exist_ok=True)

# splits = ['dev']
splits = ['train']
for split in splits:

    with open(input_path.format(split), 'r') as f:
        data = [json.loads(l) for l in f]

    question_data = get_questions(data)

    with open(os.path.join(output_dir, f'{split}_maxans_{MAX_ANS}.jsonl'), 'w') as f:
        for ex in tqdm(question_data):
            f.write(json.dumps(ex) + '\n')

    with open(os.path.join(output_dir, f'{split}_maxans_{MAX_ANS}.src'), 'w') as f:
        for ex in tqdm(question_data):
            title = ex['title']
            context = ex['context']
            answer = ex['answer']
            f.write(f'{title} {CONTEXT_TOK} {context} {ANSWER_TOK} {answer}\n')

    # with open(os.path.join(output_dir, f'{split}_og_maxans_{MAX_ANS}.src'), 'w') as f:
    #     for ex in tqdm(question_data):
    #         title = ex['title']
    #         context = ex['og_context']
    #         answer = ex['answer']
    #         f.write(f'{title} {CONTEXT_TOK} {context} {ANSWER_TOK} {answer}\n')

    # with open(os.path.join(output_dir, f'{split}_del_maxans_{MAX_ANS}.src'), 'w') as f:
    #     for ex in tqdm(question_data):
    #         title = ex['title']
    #         context = ex['del_context']
    #         answer = ex['answer']
    #         f.write(f'{title} {CONTEXT_TOK} {context} {ANSWER_TOK} {answer}\n')
