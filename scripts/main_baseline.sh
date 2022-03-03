#!/bin/bash

# for tlang in es hi de ja; do
	# bash `dirname $0`/prepare_data.sh ${tlang}

	# # train baseline adapted on combined by default
	# for direction in formal informal; do
	# 	bash `dirname $0`/baseline_finetune.sh -t ${tlang} -f ${direction} 
	# done;

	# for domain in combined; do
		# baseline
		# bash `dirname $0`/baseline_evaluate.sh -t ${tlang} -d ${domain} -m models/model.en-${tlang} -e experiments/en-${tlang}/baseline/$domain/

		# baseline adapted
		# exp_name=baseline_finetune_combined
		# for direction in formal informal; do
		# 	bash `dirname $0`/baseline_evaluate.sh -e experiments/en-${tlang}/$exp_name/${domain}/${direction}/ -t ${tlang} -d ${domain} -m experiments/en-${tlang}/$exp_name/combined/${direction}/model.en-${tlang}.adapt
		# done;

		# mbart translations
		# exp_name=mBART
		# python scripts/get_mbart_translations.py -d ${domain} -s dev -l ${tlang}
		# bash `dirname $0`/baseline_evaluate.sh -e experiments/en-${tlang}/$exp_name/${domain}/ -t ${tlang} -d ${domain} 

		# exp_name=mBART_formal
		# python scripts/finetune_mbart.py -f formal
		# python scripts/get_mbart_translations.py -d ${domain} -s dev -l ${tlang} -m models/facebook/mbart-large-50-one-to-many-mmt-finetuned-en-to-xx-formal -e ${exp_name}
		# bash `dirname $0`/baseline_evaluate.sh -e experiments/en-${tlang}/$exp_name/${domain}/ -t ${tlang} -d ${domain} 
		
		# exp_name=mBART_informal
		# python scripts/finetune_mbart.py -f informal
		# python scripts/get_mbart_translations.py -d ${domain} -s dev -l ${tlang} -m models/facebook/mbart-large-50-one-to-many-mmt-finetuned-en-to-xx-informal
		# bash `dirname $0`/baseline_evaluate.sh -e experiments/en-${tlang}/$exp_name/${domain}/ -t ${tlang} -d ${domain}

		# exp_name=mBART_formal_covariate
		# python scripts/get_mbart_translations.py -d ${domain} -s dev -l ${tlang} -m models/facebook_unfrozen/facebook/mbart-large-50-one-to-many-mmt-finetuned-covariate-en-to-xx/checkpoint-2200 -e ${exp_name} --is-covariate -f formal
		# bash `dirname $0`/baseline_evaluate.sh -e experiments/en-${tlang}/$exp_name/${domain}/ -t ${tlang} -d ${domain} 

		# exp_name=mBART_informal_covariate
		# python scripts/get_mbart_translations.py -d ${domain} -s dev -l ${tlang} -m models/facebook_unfrozen/facebook/mbart-large-50-one-to-many-mmt-finetuned-covariate-en-to-xx/checkpoint-2200 -e ${exp_name} --is-covariate -f informal
		# bash `dirname $0`/baseline_evaluate.sh -e experiments/en-${tlang}/$exp_name/${domain}/ -t ${tlang} -d ${domain} 

		# m2m100 translations
		# exp_name=m2m100_418M
		# python scripts/get_m2m_translations.py -d ${domain} -s dev -l ${tlang} -e ${exp_name}
		# bash `dirname $0`/baseline_evaluate.sh -e experiments/en-${tlang}/$exp_name/${domain}/ -t ${tlang} -d ${domain} 
		
		# exp_name=mBART_formal_${tlang}
		# python scripts/finetune_mbart.py -f formal --lang ${tlang}
		# python scripts/get_mbart_translations.py -d ${domain} -s dev -l ${tlang} -m models/facebook/mbart-large-50-one-to-many-mmt-finetuned-en-to-${tlang}-formal/checkpoint-28 -e ${exp_name}
		# bash `dirname $0`/baseline_evaluate.sh -e experiments/en-${tlang}/$exp_name/${domain}/ -t ${tlang} -d ${domain} 
		

	# done;

	# get formality scores for evaluation
	# for fname in internal_split/en-${tlang}/*; do
	# 	if [[ ${fname} != *"annotated"* ]] && [[ ${fname} != *"scores"* ]] ; then
 #  			python `dirname $0`/get_xformal_scores.py -i ${fname} -o ${fname}.finetune.scores -m /fs/clip-controllablemt/IWSLT2022/models/xformal-classifier-${tlang}
	# 	fi
	# done

	# get formality scores for evaluation
	# for fname in internal_split/en-${tlang}/*; do
	# 	if [[ ${fname} != *"annotated"* ]] && [[ ${fname} != *"scores"* ]] ; then
 #  			python `dirname $0`/get_xformal_scores.py -i ${fname} -o ${fname}.regr.scores --is-regression -m /fs/clip-controllablemt/IWSLT2022/models/xformal-regressor
	# 	fi
	# done

# done;

split=dev
slang=en
for dsplit in split0 split1 split2 split3; do
	for tlang in es hi de ja; do
		for domain in telephony topical-chat; do
			exp_name=mBART_${dsplit}
			python scripts/get_mbart_translations.py -d ${domain} -s dev -l ${tlang} --data-dir cross_val/internal_${dsplit} -e ${exp_name}
			exp_dir=experiments/en-${tlang}/$exp_name/${domain}/
			formal_ref=cross_val/internal_${dsplit}/${slang}-${tlang}/${split}.${domain}.formal.${tlang}
			informal_ref=cross_val/internal_${dsplit}/${slang}-${tlang}/${split}.${domain}.informal.${tlang}
			formal_ref_annotated=cross_val/internal_${dsplit}/${slang}-${tlang}/${split}.${domain}.formal.annotated.${tlang}
			informal_ref_annotated=cross_val/internal_${dsplit}/${slang}-${tlang}/${split}.${domain}.informal.annotated.${tlang}
			. `dirname $0`/compute_metric.sh
		done;
	done;
done;