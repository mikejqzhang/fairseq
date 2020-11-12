gpu_id=${1:-0}

model_path=/data/mjqzhang/question_generation/saved_models/nqgen_sent/checkpoint_best.pt
data_path=/data/mjqzhang/question_generation/totto_qgen-bin
date=$(date '+%m-%d-%Y')

min_len=5
max_len_const=1 # max len is computed as ax + b where x is src len
max_len_scale=1 # max len is computed as ax + b where x is src len
beam_width=5
n_hyps=10
topk=10
out_dir=/data/mjqzhang/question_generation/outputs
out_file=${out_dir}/totto_${beam_width}.tmp
mkdir -p ${out_dir}

out_file=${out_file}.txt

python summerization_generate.py ${data_path} --device-id ${gpu_id} --path ${model_path} --task summerization  --remove-bpe --gen-subset dev --batch-size 1 --min-len ${min_len} --max-len-a ${max_len_scale} --max-len-b ${max_len_const} --max-target-positions 8000 --max-source-positions 8000 --beam ${beam_width} --nbest ${n_hyps} 2>&1 | tee ${out_file}



