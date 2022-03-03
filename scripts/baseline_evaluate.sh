slang=en
tlang=hi
domain=topical-chat
exp_dir=experiments/${slang}-${tlang}/baseline/$domain/
split=dev
while getopts "t:e:d:s:m:" opt; do
  case $opt in
    t)
      tlang=$OPTARG ;;
    e)
      exp_dir=$OPTARG ;;
    d)
      domain=$OPTARG ;;
    s)
	    split=$OPTARG ;;
    m)
      model=$OPTARG ;;
    \?)
    echo "Invalid option: -$OPTARG" >&2
    exit 1 ;;
    :)
    echo "Option -$OPTARG requires an argument." >&2
    exit 1 ;;
  esac
done

data_dir=processed_data/${slang}-${tlang}
mkdir -p ${exp_dir}

if [ ! -f ${exp_dir}/out.${split} ]; then
	sockeye-translate -m $model --input ${data_dir}/${split}.${domain}.en.tok.bpe --output ${exp_dir}/out.${split}.tok.bpe
	sed -re 's/@@( |$)//g' <  ${exp_dir}/out.${split}.tok.bpe >  ${exp_dir}/out.${split}.tok
	sacremoses -l ${tlang} detokenize -x <  ${exp_dir}/out.${split}.tok >  ${exp_dir}/out.${split}
fi;

formal_ref=internal_split/${slang}-${tlang}/${split}.${domain}.formal.${tlang}
informal_ref=internal_split/${slang}-${tlang}/${split}.${domain}.informal.${tlang}
formal_ref_annotated=internal_split/${slang}-${tlang}/${split}.${domain}.formal.annotated.${tlang}
informal_ref_annotated=internal_split/${slang}-${tlang}/${split}.${domain}.informal.annotated.${tlang}
. `dirname $0`/compute_metric.sh