
eval_dir=evaluate_official
mkdir -p ${eval_dir}
for tlang in ru; do

	mkdir -p ${eval_dir}/${tlang}

	results_dir=../formality-control_UMD/EN-${tlang^^}/unconstrained/formality/blind-test

	for sid in 1 2 3 4 5; do

		outfile=${eval_dir}/${tlang}/${sid}.score
		touch $outfile

		for formality in formal informal; do
			# Formality Evaluation
			src=data/test/en-$tlang/formality-control.test.en-$tlang.en
			hyp=${results_dir}/formality-control-${sid}.${formality}.${tlang}
 			echo "Formality Score (to ${formality})" >> $outfile
 			python scripts/get_xformal_scores.py -i ${hyp} -m /fs/clip-controllablemt/IWSLT2022/models/xformal-classifier-${tlang} >> $outfile

 			# Comet Source
 			echo "Comet Score (to ${formality})" >> $outfile
 			comet-score -s ${src} -t ${hyp} --model wmt20-comet-qe-da --gpus 2 --model_storage_path /fs/clip-scratch/sweagraw/CACHE >> $outfile
		done;

		# Self-BLEU
		if [ $tlang == "ja" ]; then
			tok=ja-mecab
		else
			tok=13a
		fi;

		echo "Self-BLEU: " >> $outfile
		sacrebleu ${results_dir}/formality-control-${sid}.formal.${tlang} -tok ${tok} < ${results_dir}/formality-control-${sid}.informal.${tlang} >>  $outfile

	done;
done;

