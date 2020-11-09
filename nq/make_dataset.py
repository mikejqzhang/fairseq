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
                        start_nl_idx = document_html[start_byte:sa["start_byte"]].rfind(b'\n')
                        end_nl_idx = document_html[sa["end_byte"]:end_byte].find(b'\n')
                        if start_nl_idx != -1:
                            start_byte = start_byte + start_nl_idx + 1
                        if end_nl_idx != -1:
                            end_byte = sa["end_byte"] + end_nl_idx
                        context_html = document_html[start_byte:end_byte].decode()

                        context_soup = bs4.BeautifulSoup(context_html, "lxml")
                        for el in context_soup.find_all("a", href=True):
                            if '#cite' in el['href']:
                                el.decompose()
                        context_text = context_soup.text

                        pre_context_html = document_html[start_byte:sa["end_byte"]].decode()
                        pre_context_soup = bs4.BeautifulSoup(pre_context_html, "lxml")
                        for el in pre_context_soup.find_all("a", href=True):
                            if '#cite' in el['href']:
                                el.decompose()
                        pre_context_text = pre_context_soup.text

                        answer_html = document_html[sa["start_byte"]:sa["end_byte"]].decode()
                        answer_soup = bs4.BeautifulSoup(answer_html, "lxml")
                        for el in answer_soup.find_all("a"):
                            if '#cite' in el['href']:
                                el.decompose()
                        answer_text = answer_soup.text

                        pre_context_doc = nlp(pre_context_text)
                        context_doc = nlp(context_text)
                        pre_context_sents = [sent.text for sent in pre_context_doc.sents]
                        context_sents = [sent.text for sent in context_doc.sents]

                        if len(pre_context_sents) == 0:
                            print(document_html[sa["start_byte"]:sa["end_byte"]])
                            continue
                        context_sent_text = context_sents[len(pre_context_sents)-1]

                        if answer_text not in context_sent_text:
                            space_join = ' '.join(
                                    context_sents[len(pre_context_sents)-2:len(pre_context_sents)])
                            no_join = ''.join(
                                    context_sents[len(pre_context_sents)-2:len(pre_context_sents)])
                            if answer_text in space_join:
                                context_sent_text = space_join
                            elif answer_text in no_join:
                                context_sent_text = no_join
                            else:
                                print(pre_context_sents)
                                print(context_sents)
                                print(context_sent_text)
                                print(answer_text)
                                print(question_text)
                                print('='*80)
                                continue
                        if '\n' in context_sent_text:
                            print(pre_context_sents)
                            print(context_sents)
                            print('-'*80)
                            print(context_sent_text)
                            print('-'*80)
                            print(answer_text)
                            print(question_text)
                            print('='*80)
                            continue


                        sent_ex = {'question':question_text,
                                   'context': context_sent_text.strip(),
                                   'title': doc_title,
                                   'answer': answer_text}
                        block_ex = {'question':question_text,
                                    'context': context_text.strip(),
                                    'title': doc_title,
                                    'answer': answer_text}
                        if sent_ex not in sent_data and answer_text in context_sent_text:
                            sent_data.append(sent_ex)
                        if block_ex not in block_data:
                            block_data.append(block_ex)
    return sent_data, block_data


if __name__ == '__main__':
    # splits = ['train', 'dev']
    splits = ['sample']
    num_threads = 12

    for split in splits:
        input_pattern = f'/data/mjqzhang/original_nq/v1.0/{split}/*'
        sent_dir = f'/data/mjqzhang/question_generation/nqgen_sent'
        block_dir = f'/data/mjqzhang/question_generation/nqgen_block'
        os.makedirs(sent_dir, exist_ok=True)
        os.makedirs(block_dir, exist_ok=True)


        input_paths = glob(input_pattern)

        # sent_data, block_data = _convert_qa_to_qgen(input_paths[0])

        # sent_block_shards = [_convert_qa_to_qgen(ip) for ip in input_paths]
        # sent_data, block_data = zip(*sent_block_shards)
        # sent_data = [ex for shard in sent_data for ex in shard]
        # block_data = [ex for shard in block_data for ex in shard]

        pool = multiprocessing.Pool(num_threads)
        sent_block_shards = pool.map(_convert_qa_to_qgen, input_paths)
        sent_data, block_data = zip(*sent_block_shards)
        sent_data = [ex for shard in sent_data for ex in shard]
        block_data = [ex for shard in block_data for ex in shard]


        with open(os.path.join(sent_dir, f'{split}.src'), 'w') as sent_src_f, \
                open(os.path.join(sent_dir, f'{split}.tgt'), 'w') as sent_tgt_f:
            for i, ex in enumerate(sent_data):
                question = ex['question']
                context = ex['context']
                title = ex['title']
                answer = ex['answer']
                sent_src_f.write(f'{title} {CONTEXT_TOK} {context} {ANSWER_TOK} {answer}\n')
                sent_tgt_f.write(f'{question}\n')

        with open(os.path.join(block_dir, f'{split}.src'), 'w') as block_src_f, \
                open(os.path.join(block_dir, f'{split}.tgt'), 'w') as block_tgt_f:
            for ex in sent_data:
                question = ex['question']
                context = ex['context']
                title = ex['title']
                answer = ex['answer']
                block_src_f.write(f'{title} {CONTEXT_TOK} {context} {ANSWER_TOK} {answer}\n')
                block_tgt_f.write(f'{question}\n')
