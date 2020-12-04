import json
import torch
from tqdm import tqdm
from fairseq.models.bart import BARTModel

# source_file = '/data/mjqzhang/question_generation/totto_qgen/v2/dev_maxans_3.src'
# output_file = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_maxans_3.hyp'
# source_file = '/data/mjqzhang/question_generation/totto_qgen/v2/dev_og_maxans_3.src'
# output_file = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_og_maxans_3.hyp'
# source_file = '/data/mjqzhang/question_generation/totto_qgen/v2/dev_del_maxans_3.src'
# output_file = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_del_maxans_3.hyp'

# split = 3
# print(f'Running Split: {split}')

bart = BARTModel.from_pretrained(
	'/data/mjqzhang/question_generation/saved_models/nqgen_sent_v2',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/data/mjqzhang/question_generation/nqgen_sent_v2-bin'
)


# source_file = '/data/mjqzhang/question_generation/totto_qgen/v2/train_ans_in_context.src'
# output_file = f'/data/mjqzhang/question_generation/totto_qgen/outputs/train_ans_in_context_s{split}.hyp'
source_file = '/data/mjqzhang/question_generation/totto_qgen/v2/dev_ans_in_context.src'
output_file = '/data/mjqzhang/question_generation/totto_qgen/outputs/dev_ans_in_context.hyp'


# device = torch.device(f'cuda:{split}')
# bart.to(device=device)
bart.cuda()
bart.eval()
bart.half()

BATCH_SIZE = 64
# n_batches_per_split = 305436 // (4 * BATCH_SIZE)
# ind_lower = split * n_batches_per_split * BATCH_SIZE
# ind_upper = (split+1) * n_batches_per_split * BATCH_SIZE if split != 3 else 305436

with open(source_file, 'r') as fsrc:
    source_data = [l.strip() for l in fsrc]
    # source_data = [l.strip() for i, l in enumerate(fsrc) if i >= ind_lower and i < ind_upper]

with open(output_file, 'w') as fout:
    with torch.no_grad():
        for i in tqdm(range(0, len(source_data), BATCH_SIZE)):
            batched_samples = bart.sample_nbest(
                    source_data[i:i+BATCH_SIZE], beam=10, min_len=3, max_len=60,
                    nbest=10, lenpen=1.0, no_repeat_ngram_size=3)
            for sample in batched_samples:
                fout.write(json.dumps(sample) + '\n')
                fout.flush()
