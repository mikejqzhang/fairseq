import torch
from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained(
	'/data/mjqzhang/question_generation/saved_models/nqgen_sent/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/data/mjqzhang/question_generation/nqgen_sent-bin'
)
source_file = '/data/mjqzhang/question_generation/totto_qgen_head/head.src'
output_file = '/data/mjqzhang/question_generation/outputs/totto_qgen_head.out'

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 1

with open(source_file) as source, open(output_file, 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            print(slines)
            with torch.no_grad():
                hypotheses_batch = bart.sample(
                        slines,
                        beam=10,
                        min_len=5,
                        max_len_a=1,
                        max_len_b=1,
                        max_target_positions=8000,
                        max_source_positions=8000,
                        nbest=10,
                        )

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
