slang=en
tlang=hi
domain=topical-chat
exp_dir=experiments/${slang}-${tlang}/baseline/$domain/
split=dev
input=None
generic=False
preprocess=False
while getopts "t:e:d:s:m:i:gp" opt; do
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
		i)
			input=$OPTARG ;;
		g)
			generic=True ;;
		p)
			preprocess=True ;;
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

	if [ $preprocess == True ]; then
		echo "Preprocessing $input and storing to ${exp_dir}/input.tok.bpe"
		sacremoses -l en tokenize -x < ${input}> ${exp_dir}/input.tok
		subword-nmt apply-bpe --codes models/model.en-${tlang}/bpe_codes < ${exp_dir}/input.tok > ${exp_dir}/input.tok.bpe
	else
		cp $input ${exp_dir}/input.tok.bpe
	fi;

	echo "Translating ${exp_dir}/input.tok.bpe"
	sockeye-translate -m $model --input ${exp_dir}/input.tok.bpe --output ${exp_dir}/out.${split}.tok.bpe
	sed -re 's/@@( |$)//g' <  ${exp_dir}/out.${split}.tok.bpe >  ${exp_dir}/out.${split}.tok
	sacremoses -l ${tlang} detokenize -x <  ${exp_dir}/out.${split}.tok >  ${exp_dir}/out.${split}
fi;

if [ $generic == False ]; then
	echo "Running Task Evaluation on ${exp_dir}/out.${split}"
	formal_ref=internal_split/${slang}-${tlang}/${split}.${domain}.formal.${tlang}
	informal_ref=internal_split/${slang}-${tlang}/${split}.${domain}.informal.${tlang}
	formal_ref_annotated=internal_split/${slang}-${tlang}/${split}.${domain}.formal.annotated.${tlang}
	informal_ref_annotated=internal_split/${slang}-${tlang}/${split}.${domain}.informal.annotated.${tlang}
	. `dirname $0`/compute_metric.sh
else
	echo "Running Generic Evaluation on ${exp_dir}/out.${split}"
	hyp=${exp_dir}/out.${split}
	src=${input}
	if [ $tlang != hi ] || [ $tlang != ja ]; then
		ref=../mustc/en-$tlang/data/tst-COMMON/txt/*.${tlang}
	else
		ref=../mustc/en-$tlang/*.${tlang}
	fi;

	if [ $tlang == "ja" ]; then
			tok=ja-mecab
		else
			tok=13a
		fi;
		echo "BLEU: " > $exp_dir/scores
		sacrebleu ${hyp} -tok ${tok} < ${ref} >> $exp_dir/scores

		echo "COMET: " >> $exp_dir/scores
		comet-score -s ${src} -t ${hyp} -r ${ref} --gpus 2 --model_storage_path /fs/clip-scratch/sweagraw/CACHE >> $exp_dir/scores

fi;