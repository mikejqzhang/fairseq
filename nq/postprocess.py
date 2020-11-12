import json
import argparse

def postprocess(args):
    vocab_path = '/data/mjqzhang/question_generation/bart/encoder.json'
    with open(vocab_path) as f:
        tok2idx = json.load(f)
    idx2tok = {idx: tok for tok, idx in tok2idx.items()}
    with open(args.hyp_path) as inp_f, open(args.hyp_path + '.post', 'w') as out_f:
        for line in inp_f:
            post_line = [idx2tok[int(idx)] for idx in line.split()]
            out_f.write(' '.join(post_line) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('hyp_path', type=str,
                        help='an integer for the accumulator')
    args = parser.parse_args()
    postprocess(args)
