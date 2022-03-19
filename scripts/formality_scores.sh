# python `dirname $0`/get_xformal_scores.py -i ../synthetic/de_en_formal.tsv -o ../synthetic/de_en_formal.tsv.scores -m /fs/clip-controllablemt/IWSLT2022/models/xformal-classifier-de
# python `dirname $0`/get_xformal_scores.py -i ../synthetic/de_en_informal.tsv -o ../synthetic/de_en_informal.tsv.scores -m /fs/clip-controllablemt/IWSLT2022/models/xformal-classifier-de
# python `dirname $0`/get_xformal_scores.py -i ../synthetic/es_en_formal.tsv -o ../synthetic/es_en_formal.tsv.scores -m /fs/clip-controllablemt/IWSLT2022/models/xformal-classifier-es
# python `dirname $0`/get_xformal_scores.py -i ../synthetic/es_en_informal.tsv -o ../synthetic/es_en_informal.tsv.scores -m /fs/clip-controllablemt/IWSLT2022/models/xformal-classifier-es

# for tlang in es ja de hi; do
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


# for fname in ../model_outputs/finetuned_paired_res/*; do
# 	if [[ ${fname} != *".out"* ]]  && [[ ${fname} != *".sh"* ]] ; then
# 		echo $fname
# 		tlang=${fname: -2}
#  			python `dirname $0`/get_xformal_scores.py -i ${fname} -o ${fname}.scores -m /fs/clip-controllablemt/IWSLT2022/models/xformal-classifier-${tlang}
# 	fi
# done

# for fname in ../model_outputs/finetuned_unpaired_res/*; do
# 	if [[ ${fname} != *".out"* ]]  && [[ ${fname} != *".sh"* ]] ; then
# 		echo $fname
# 		tlang=${fname: -2}
#  			python `dirname $0`/get_xformal_scores.py -i ${fname} -o ${fname}.scores -m /fs/clip-controllablemt/IWSLT2022/models/xformal-classifier-${tlang}
# 	fi
# done


for fname in ../model_outputs/finetuned-res/*; do
	if [[ ${fname} != *".scores"* ]] && [[ ${fname} != *".out"* ]]  && [[ ${fname} != *".sh"* ]] ; then
		echo $fname
		tlang=${fname: -2}
 			python `dirname $0`/get_xformal_scores.py -i ${fname} -o ${fname}.scores -m /fs/clip-controllablemt/IWSLT2022/models/xformal-classifier-${tlang}
	fi
done

for fname in ../model_outputs/pretrained-res/*; do
	if [[ ${fname} != *".scores"* ]] && [[ ${fname} != *".out"* ]]  && [[ ${fname} != *".sh"* ]] ; then
		echo $fname
		tlang=${fname: -2}
 			python `dirname $0`/get_xformal_scores.py -i ${fname} -o ${fname}.scores -m /fs/clip-controllablemt/IWSLT2022/models/xformal-classifier-${tlang}
	fi
done