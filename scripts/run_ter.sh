scripts_path=/fs/clip-scratch/sweagraw/Editor/scripts
TER="/fs/clip-scratch/sweagraw/software/tercom/tercom-0.10.0.jar"

slang=en
domain=combined
split=train

for tlang in hi es ja de; do

	ref=processed_data/${slang}-${tlang}/${split}.${domain}.formal.${tlang}.tok.bpe
	hyp=processed_data/${slang}-${tlang}/${split}.${domain}.informal.${tlang}.tok.bpe

	out_dir=ter_dist/${slang}-${tlang}/${split}.${domain}.${tlang}/

	mkdir -p $out_dir

	python ${scripts_path}/add_sen_id.py ${ref} ${out_dir}/ref
	python ${scripts_path}/add_sen_id.py ${hyp} ${out_dir}/hyp

	java -jar $TER -r ${out_dir}/ref -h ${out_dir}/hyp -n ${out_dir}/out -s

done;

for tlang in hi es ja de; do
	echo ${tlang}
	tail -n 1 ter_dist/en-${tlang}/${split}.combined.${tlang}/out.sum
done;