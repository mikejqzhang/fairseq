import csv
import json
import spacy
from tqdm import tqdm

src_file = '/data/mjqzhang/question_generation/totto_qgen/v2/dev_maxans_3.jsonl'
hyp_file = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_maxans_3.hyp'

with open(src_file) as f:
    data = [json.loads(l) for l in f]

with open(hyp_file) as f:
    for ex, hyps in zip(data, [json.loads(l) for l in f]):
        ex['hyps'] = hyps

nlp = spacy.load("en_core_web_sm")

def get_entities(sent):
    sent = nlp(sent)

output_rows = []
for ex in tqdm(data):
    # src_ents = set(x.text.lower() for x in nlp(ex['context']).ents) | set(x.text.lower() for x in nlp(ex['title']).ents)
    context_nouns = set(x.text.lower() for x in nlp(ex['context']) if x.pos_ in ['PROPN', 'NOUN', 'NUM'])
    title_nouns = set(x.text.lower() for x in nlp(ex['title']) if x.pos_ in ['PROPN', 'NOUN', 'NUM'])
    src_nouns = context_nouns | title_nouns
    ex['best_hyp'] = None

    for hyp in ex['hyps']:
        # hyp_ents = set(x.text.lower() for x in nlp(hyp).ents)
        hyp_nouns = set(x.text.lower() for x in nlp(hyp) if x.pos_ in ['PROPN', 'NOUN', 'NUM'])
        if len(hyp_nouns - src_nouns) == 0:
            ex['best_hyp'] = hyp
            break
    if ex['best_hyp'] != None:
        output_rows.append([ex['title'], ex['context'], ex['answer'], ex['best_hyp']])

with open('temp.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(output_rows)
