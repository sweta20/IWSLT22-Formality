
if [ $tlang == "ja" ]; then
  sacrebleu ${formal_ref} -tok char < ${exp_dir}/out.${split} > ${exp_dir}/scores
  sacrebleu ${informal_ref} -tok char < ${exp_dir}/out.${split} >> ${exp_dir}/scores
    python scorer.py \
  		-hyp ${exp_dir}/out.${split} \
  		-f  ${formal_ref_annotated} \
  		-if ${informal_ref_annotated} -nd >> ${exp_dir}/scores
else
  sacrebleu ${formal_ref} < ${exp_dir}/out.${split} > ${exp_dir}/scores
  sacrebleu ${informal_ref} < ${exp_dir}/out.${split} >> ${exp_dir}/scores
  python scorer.py \
      -hyp ${exp_dir}/out.${split} \
      -f  ${formal_ref_annotated} \
      -if ${informal_ref_annotated}>> ${exp_dir}/scores
fi;
cat  ${exp_dir}/scores
