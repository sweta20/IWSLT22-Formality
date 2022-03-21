# offical eval

slang=en
for tlang in es de ru it hi ja; do
	out_data_dir=processed_data/${slang}-${tlang}
	mkdir -p data_dir
	for formality in formal informal; do
		src=data/test/en-${tlang}/formality-control.test.en-$tlang.en
		echo "Preprocessing $src"
		sacremoses -l en tokenize -x < ${src} > $out_data_dir/test.official.en.tok
		subword-nmt apply-bpe --codes models/model.en-${tlang}/bpe_codes < $out_data_dir/test.official.en.tok > $out_data_dir/test.official.en.tok.bpe
	done;

	src=$out_data_dir/test.official.en.tok.bpe
	split=test

	# baseline
	exp_dir=experiments/en-$tlang/baseline/${split}
	mkdir -p $exp_dir
	sockeye-translate -m models/model.en-${tlang} --input $src --output ${exp_dir}/out.${split}.tok.bpe
	sed -re 's/@@( |$)//g' <  ${exp_dir}/out.${split}.tok.bpe >  ${exp_dir}/out.${split}.tok
	sacremoses -l ${tlang} detokenize -x <  ${exp_dir}/out.${split}.tok >  ${exp_dir}/out.${split}

	# baseline-finetuned
	exp_name=baseline_finetune_combined
	for direction in formal informal; do
		# baseline
		exp_dir=experiments/en-$tlang/${exp_name}/${split}/${direction}
		mkdir -p $exp_dir
		sockeye-translate -m experiments/en-${tlang}/$exp_name/combined/${direction}/model.en-${tlang}.adapt --input $src --output ${exp_dir}/out.${split}.tok.bpe
		sed -re 's/@@( |$)//g' <  ${exp_dir}/out.${split}.tok.bpe >  ${exp_dir}/out.${split}.tok
		sacremoses -l ${tlang} detokenize -x <  ${exp_dir}/out.${split}.tok >  ${exp_dir}/out.${split}
	done;

done;


