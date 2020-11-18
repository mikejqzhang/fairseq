import json
import argparse
import torch
from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained(
	'/data/mjqzhang/question_generation/saved_models/nqgen_sent_v2',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/data/mjqzhang/question_generation/nqgen_sent_v2-bin'
)

def postprocess(args):
    with open(args.hyp_path) as inp_f, open(args.hyp_path + '.post', 'w') as out_f:
        for line in inp_f:
            post_line = bart.decode(line)
            out_f.write(' '.join(post_line) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('hyp_path', type=str,
                        help='an integer for the accumulator')
    args = parser.parse_args()
    postprocess(args)
