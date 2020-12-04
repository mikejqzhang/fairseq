import os
import gzip
import json
import multiprocessing

import bs4
from glob import glob
from spacy.lang.en import English

MAX_SA_TOKENS = 5

CONTEXT_TOK = '[CTX]'
ANSWER_TOK = '[ANS]'

def _convert_qa_to_qgen(input_path):
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    sent_data = []
    skipped_data = []
    with gzip.open(input_path) as input_file:
        for line in input_file:
            json_example = json.loads(line)
            question_text = json_example["question_text"]
            document_html = json_example["document_html"].encode("utf-8")
            doc_tokens = json_example['document_tokens']
            doc_title = json_example['document_title']

            for annotation in json_example["annotations"]:
                for sa in annotation["short_answers"]:
                    if sa["end_token"] - sa["start_token"] <= MAX_SA_TOKENS:
                        context_start = sa["start_token"]
                        while not doc_tokens[context_start]['html_token']:
                            context_start -= 1

                        context_end = sa["end_token"]
                        while not doc_tokens[context_end]['html_token']:
                            context_end += 1

                        if doc_tokens[context_start]['token'] != '<P>':
                            continue
                        soup = bs4.BeautifulSoup(
                                ' '.join([x['token'] for x in doc_tokens[:context_end+1]]),
                                "lxml")
                        answer_el = soup.find_all('p')[-1]
                        answer_parents = [el.name for el in answer_el.parents]
                        if answer_parents != ['body', 'html', '[document]']:
                            # print(answer_parents)
                            # print('='*80)
                            continue

                        start_byte = doc_tokens[context_start]["start_byte"]
                        end_byte = doc_tokens[context_end]["end_byte"]

                        # start_nl_idx = document_html[start_byte:sa["start_byte"]].rfind(b'\n')
                        # end_nl_idx = document_html[sa["end_byte"]:end_byte].find(b'\n')
                        # if start_nl_idx != -1:
                        #     start_byte = start_byte + start_nl_idx + 1
                        # if end_nl_idx != -1:
                        #     end_byte = sa["end_byte"] + end_nl_idx

                        context_html = document_html[start_byte:end_byte].decode()
                        context_soup = bs4.BeautifulSoup(context_html, "lxml")
                        for el in context_soup.find_all("a", href=True):
                            if '#cite' in el['href']:
                                el.decompose()
                        context_text = context_soup.text.replace('\n', ' ').strip()

                        pre_context_html = document_html[start_byte:sa["end_byte"]].decode()
                        pre_context_soup = bs4.BeautifulSoup(pre_context_html, "lxml")
                        for el in pre_context_soup.find_all("a", href=True):
                            if '#cite' in el['href']:
                                el.decompose()
                        pre_context_text = pre_context_soup.text.replace('\n', ' ').strip()

                        answer_html = document_html[sa["start_byte"]:sa["end_byte"]].decode()
                        answer_soup = bs4.BeautifulSoup(answer_html, "lxml")
                        for el in answer_soup.find_all("a"):
                            if '#cite' in el['href']:
                                el.decompose()
                        answer_text = answer_soup.text.replace('\n', ' ').strip()

                        pre_context_doc = nlp(pre_context_text)
                        context_doc = nlp(context_text)
                        pre_context_sents = [sent.text for sent in pre_context_doc.sents]
                        context_sents = [sent.text for sent in context_doc.sents]

                        context_sent_text = context_sents[len(pre_context_sents)-1]
                        sent_ex = {'question':question_text,
                                   'context': context_sent_text,
                                   'title': doc_title,
                                   'answer': answer_text}

                        if answer_text not in context_sent_text:
                            skipped_data.append(sent_ex)
                        else:
                            sent_data.append(sent_ex)
    return sent_data, skipped_data


if __name__ == '__main__':
    # splits = ['train', 'dev']
    splits = ['sample']
    # splits = ['dev']
    num_threads = 12

    for split in splits:
        input_pattern = f'/data/mjqzhang/original_nq/v1.0/{split}/*'
        sent_dir = f'/data/mjqzhang/question_generation/nqgen_sent_v2'
        os.makedirs(sent_dir, exist_ok=True)

        input_paths = glob(input_pattern)

        pool = multiprocessing.Pool(num_threads)
        sent_block_shards = pool.map(_convert_qa_to_qgen, input_paths)
        sent_data, skipped_data = zip(*sent_block_shards)
        skipped_data = [ex for shard in skipped_data for ex in shard]
        sent_data = [ex for shard in sent_data for ex in shard]

        print('Skipped {} over {} examples and kept {} out of {}'.format(
            split, len(skipped_data), len(sent_data), len(skipped_data) + len(sent_data)))

        # with open(os.path.join(sent_dir, f'{split}.jsonl'), 'w') as f:
        #     for ex in sent_data:
        #         f.write(json.dumps(ex) + '\n')
        # with open(os.path.join(sent_dir, f'{split}.skip.jsonl'), 'w') as f:
        #     for ex in skipped_data:
        #         f.write(json.dumps(ex) + '\n')
