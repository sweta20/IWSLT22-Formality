#!/bin/bash

# SBATCH --job-name=xformal
# SBATCH --time=1-00:00:00
# SBATCH --mem=40g
# SBATCH --qos=gpu-medium
# SBATCH --exclude=materialgpu00,materialgpu02
# SBATCH --nodelist=clipgpu01
# SBATCH --cpus-per-task=6
# SBATCH --gres=gpu:1
# SBATCH --partition=gpu


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fs/clip-xling/projects/semdiv/anaconda3/lib

source ~/.bashrc

conda activate /fs/clip-xling/projects/semdiv/anaconda3/envs/xformal
export TRANSFORMERS_CACHE=/fs/clip-scratch/sweagraw/CACHE
mkdir -p $TRANSFORMERS_CACHE

base_model="xlm-roberta-base"
#args=" --finetune "
args=""
learning_rate="5e-5"
epoch="3"

tune_task=binary_classification
output_dir=/fs/clip-controllablemt/IWSLT2022/models
run_dir=/fs/clip-controllablemt/IWSLT2022/runs

for tune_lang in ru; do
  if [[ $tune_task == binary_classification ]]; then
    # data_dir=/fs/clip-xling/projects/xformal/data/formality_classifiers_data/$tune_lang
    data_dir="/fs/clip-controllablemt/IWSLT2022/formality_classifiers_data/$tune_lang"
  else
    data_dir=/fs/clip-xling/projects/xformal/data/formality_regression_data/$tune_lang
  fi

  # tune -> dev
  if [ ! -d $output_dir/xformal-classifier-${tune_lang} ]; then
		python scripts/run_gyafc.py \
  			--model_name_or_path $base_model \
  			--do_train \
  			--do_eval \
  			--train_file $data_dir/train.csv \
  			--validation_file $data_dir/dev.csv \
  			--max_seq_length 128 \
  			--per_device_train_batch_size 32 \
  			--learning_rate $learning_rate \
  			--num_train_epochs $epoch \
			  --logging_dir $run_dir \
 	  		--cache_dir $TRANSFORMERS_CACHE \
  			--output_dir $output_dir/xformal-classifier-${tune_lang}/ \
        ${args}
  fi;
done
