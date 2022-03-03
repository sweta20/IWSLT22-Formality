
exp_dir=experiments
slang=en
tlang=de
exp_name=baseline_finetune_combined
direction=formal
domain=combined
split=dev
while getopts "s:t:e:f:d:" opt; do
  case $opt in
    s)
      slang=$OPTARG ;;
    t)
      tlang=$OPTARG ;;
    e)
      exp_name=$OPTARG ;;
    f)
      direction=$OPTARG ;;
    d)
      domain=$OPTARG ;;
    \?)
    echo "Invalid option: -$OPTARG" >&2
    exit 1 ;;
    :)
    echo "Option -$OPTARG requires an argument." >&2
    exit 1 ;;
  esac
done

data_dir=processed_data/${slang}-${tlang}
sub_exp_dir=${exp_dir}/${slang}-${tlang}/$exp_name/${domain}/${direction}

mkdir -p $sub_exp_dir

sockeye-prepare-data \
    --source $data_dir/train.${domain}.en.tok.bpe --target $data_dir/train.${domain}.${direction}.${tlang}.tok.bpe \
    --source-vocab models/model.en-${tlang}/vocab.src.0.json --target-vocab models/model.en-${tlang}/vocab.trg.0.json --shared-vocab \
    --output $sub_exp_dir/data 

torchrun --no_python --nproc_per_node 1 sockeye-train --config models/model.en-${tlang}/args.yaml \
--prepared-data $sub_exp_dir/data --validation-source ${data_dir}/dev.${domain}.en.tok.bpe \
 --validation-target ${data_dir}/dev.${domain}.${direction}.${tlang}.tok.bpe \
  --output ${sub_exp_dir}/model.en-${tlang}.adapt --params models/model.en-${tlang}/params.best \
  --learning-rate-scheduler none --initial-learning-rate 0.00001 --batch-size 1024 \
  --update-interval 1 --checkpoint-interval 10 --max-num-checkpoint-not-improved 5
