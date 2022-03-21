slang=en
tlang=hi
domain=topical-chat
exp_dir=experiments/${slang}-${tlang}/baseline/$domain/
split=dev
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
		g)
			generic=True ;;
		\?)
		echo "Invalid option: -$OPTARG" >&2
		exit 1 ;;
		:)
		echo "Option -$OPTARG requires an argument." >&2
		exit 1 ;;
	esac
done


mkdir -p ${exp_dir}

if [ $generic == True ]; then
	if [ $tlang != hi ] || [ $tlang != ja ]; then
		src=../mustc/en-$tlang/data/tst-COMMON/txt/*.en
	else
		src=../mustc/en-$tlang/*.en
	fi;
else
	src=internal_split/${slang}-${tlang}/dev.combined.en
fi;

if [ ! -f ${exp_dir}/input.tok.bpe ]; then
	echo "Preprocessing ${exp_dir}/input.tok.bpe"
	sacremoses -l en tokenize -x < ${src} > ${exp_dir}/input.tok
	subword-nmt apply-bpe --codes models/model.en-${tlang}/bpe_codes < ${exp_dir}/input.tok > ${exp_dir}/input.tok.bpe
fi;

if [ ! -f ${exp_dir}/out.${split} ]; then
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