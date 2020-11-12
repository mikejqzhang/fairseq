import os
import gzip
import json
import multiprocessing

import bs4
from glob import glob
from spacy.lang.en import English

def _convert_qa_to_qgen(input_path):
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    filtered_questions = []
    with gzip.open(input_path) as input_file:
        for line in input_file:
            json_example = json.loads(line)
            question_text = json_example["question_text"]
            document_html = json_example["document_html"].encode("utf-8")
            doc_tokens = json_example['document_tokens']
            doc_title = json_example['document_title']

            in_table = False
            for annotation in json_example["annotations"]:
                if in_table:
                    break
                for sa in annotation["short_answers"]:
                    context_start = sa["start_token"]
                    while not doc_tokens[context_start]['html_token']:
                        context_start -= 1
                    if doc_tokens[context_start]['token'] != '<Td>':
                        continue
                    context_end = sa["end_token"]

                    soup = bs4.BeautifulSoup(
                            ' '.join([x['token'] for x in doc_tokens[:context_end+1]]),
                            "lxml")
                    answer_el = soup.find_all('td')[-1]
                    answer_parents = [el.name for el in answer_el.parents]
                    if 'table' in answer_parents:
                        in_table = True
                        break
            if in_table:
                filtered_questions.append(question_text)

    return filtered_questions


if __name__ == '__main__':
    # splits = ['train', 'dev']
    # splits = ['sample']
    splits = ['train']
    # splits = ['dev']
    num_threads = 24

    for split in splits:
        input_pattern = f'/data/mjqzhang/original_nq/v1.0/{split}/*'
        output_path = f'/data/mjqzhang/question_generation/nq_filtered.{split}.txt'
        input_paths = glob(input_pattern)

        # filtered_question_shards = [_convert_qa_to_qgen(ip) for ip in input_paths]
        # filtered_questions = [ex for shard in filtered_question_shards for ex in shard]

        pool = multiprocessing.Pool(num_threads)
        filtered_question_shards = pool.map(_convert_qa_to_qgen, input_paths)
        filtered_questions = [ex for shard in filtered_question_shards for ex in shard]

        with open(output_path, 'w') as f:
            for question in filtered_questions:
                f.write(f'{question}\n')
