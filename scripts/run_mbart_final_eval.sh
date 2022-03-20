
touch generic_mbart_scores

# for tlang in es de ru it; do
# 	echo "Target Language: $tlang" >> generic_mbart_scores
# 	for formality in formal informal neutral; do
# 		echo "Formality: $formality" >> generic_mbart_scores
# 		exp_name=mBART_${formality}_covariate_all

# 		src=../mustc/en-$tlang/data/tst-COMMON/txt/*.en
# 		ref=../mustc/en-$tlang/data/tst-COMMON/txt/*.${tlang}

# 		# python scripts/get_mbart_translations.py -l $tlang --source ${src} -m models/vastai/covariate-alldata -e ${exp_name} --is-covariate -f ${formality} -s generic 
		
# 		hyp=experiments/en-${tlang}/${exp_name}/None/out.generic

# 		echo "BLEU: " >> generic_mbart_scores
# 		sacrebleu ${hyp} -tok 13a < ${ref} >> $generic_mbart_scores

# 		echo "COMET: " >> generic_mbart_scores
# 		comet-score -s ${src} -t ${hyp} -r ${ref} --gpus 0 >> $generic_mbart_scores
# 	done;
# done;


for tlang in hi; do
	echo "Target Language: $tlang" >> generic_mbart_scores
	for formality in formal informal neutral; do
		echo "Formality: $formality" >> generic_mbart_scores
		exp_name=mBART_${formality}_covariate_all

		src=../mustc/en-$tlang/*.en
		ref=../mustc/en-$tlang/*.${tlang}

		# python scripts/get_mbart_translations.py -l $tlang --source ${src} -m models/vastai/covariate-alldata -e ${exp_name} --is-covariate -f ${formality} -s generic 

		hyp=experiments/en-${tlang}/${exp_name}/None/out.generic

		if [ $tlang == "ja" ]; then
			tok=ja-mecab
		else
			tok=13a
		fi;

		echo "BLEU: " >> generic_mbart_scores
		sacrebleu ${hyp} -tok ${tok} < ${ref} >> $generic_mbart_scores

		echo "COMET: " >> generic_mbart_scores
		comet-score -s ${src} -t ${hyp} -r ${ref} --gpus 0 >> $generic_mbart_scores
	done;
done;


# # offical eval
# for tlang in es de ru it hi ja; do
# 	for formality in formal informal; do
# 		exp_name=mBART_formal_covariate_all
# 		python scripts/get_mbart_translations.py -l $tlang --source data/test/en-$tlang/formality-control.test.en-$tlang.en -m models/vastai/covariate-alldata -e ${exp_name} --is-covariate -f ${formality} -s blind_test --output ../formality-control_UMD/EN-${tlang^^}/unconstrained/formality/blind-test/formality-control-4.${formality}.${tlang}
# 	done;
# done;
