import gzip
import json
import multiprocessing

import bs4
from glob import glob
from spacy.lang.en import English

MAX_TOKENS=5

def _convert_qa_to_qgen(input_path):
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    sent_data = []
    block_data = []
    with gzip.open(input_path) as input_file:
        for line in input_file:
            json_example = json.loads(line)
            question_text = json_example["question_text"]
            document_html = json_example["document_html"].encode("utf-8")
            doc_tokens = json_example['document_tokens']
            doc_title = json_example['document_title']

            for annotation in json_example["annotations"]:
                for sa in annotation["short_answers"]:
                    if sa["end_token"] - sa["start_token"] <= MAX_TOKENS:
                        context_start = sa["start_token"]
                        while not doc_tokens[context_start]['html_token']:
                            context_start -= 1

                        if doc_tokens[context_start]['token'] != '<P>':
                            continue

                        context_end = sa["end_token"]
                        while not doc_tokens[context_end]['html_token']:
                            context_end += 1

                        context_html = document_html[doc_tokens[context_start]["start_byte"]:doc_tokens[context_end]["end_byte"]].decode()
                        context_soup = bs4.BeautifulSoup(context_html, "lxml")
                        for el in context_soup.find_all("a", href=True):
                            if '#cite' in el['href']:
                                el.decompose()
                        context_text = context_soup.text

                        pre_context_html = document_html[doc_tokens[context_start]["start_byte"]:sa["start_byte"]].decode()
                        pre_context_soup = bs4.BeautifulSoup(pre_context_html, "lxml")
                        for el in pre_context_soup.find_all("a", href=True):
                            if '#cite' in el['href']:
                                el.decompose()
                        pre_context_text = pre_context_soup.text

                        answer_html = document_html[sa["start_byte"]:sa["end_byte"]].decode()
                        answer_soup = bs4.BeautifulSoup(answer_html, "lxml")
                        for el in answer_soup.find_all("a"):
                            if '#cite_note' in el['href']:
                                el.decompose()
                        answer_text = answer_soup.text
                        
                        pre_context_doc = nlp(pre_context_text)
                        context_doc = nlp(context_text)

                        pre_context_sents = [sent.text for sent in pre_context_doc.sents]
                        context_sents = [sent.text for sent in context_doc.sents]

                        if len(pre_context_sents) == 0:
                            context_sent_text = context_sents[0]
                        elif pre_context_sents[-1] != context_sents[len(pre_context_sents)-1]:
                            context_sent_text = context_sents[len(pre_context_sents)-1]
                        else:
                            context_sent_text = context_sents[len(pre_context_sents)]

                        if answer_text not in context_sent_text:
                            print(pre_context_sents)
                            print(context_sents)
                            print(context_sent_text)
                            print(answer_text)
                            print(question_text)
                            print('='*80)
                            
                        sent_ex = {'question':question_text,
                                   'context': context_sent_text,
                                   'title': doc_title,
                                   'answer': answer_text}
                        block_ex = {'question':question_text,
                                    'context': context_text,
                                    'title': doc_title,
                                    'answer': answer_text}
                        if sent_ex not in sent_data and answer_text in context_sent_text:
                            sent_data.append(sent_ex)
                        if block_ex not in block_data:
                            block_data.append(block_ex)
    return sent_data, block_data


if __name__ == '__main__':
    splits = ['train', 'dev']
    # splits = ['sample']
    for split in splits:
        input_pattern = f'/data/mjqzhang/original_nq/v1.0/{split}/*'
        sent_out_path = f'/data/mjqzhang/question_generation/nq_qgen/{split}.sent.jsonl'
        block_out_path = f'/data/mjqzhang/question_generation/nq_qgen/{split}.block.jsonl'
        num_threads = 12

        input_paths = glob(input_pattern)

        # sent_data, block_data = _convert_qa_to_qgen(input_paths[0])

        pool = multiprocessing.Pool(num_threads)
        sent_block_shards = pool.map(_convert_qa_to_qgen, input_paths)
        sent_data, block_data = zip(*sent_block_shards)
        sent_data = [ex for shard in sent_data for ex in shard]
        block_data = [ex for shard in block_data for ex in shard]


        with open(sent_out_path, 'w') as output_file:
            for ex in sent_data:
                output_file.write(json.dumps(ex))
                output_file.write('\n')
        with open(block_out_path, 'w') as output_file:
            for ex in block_data:
                output_file.write(json.dumps(ex))
                output_file.write('\n')
