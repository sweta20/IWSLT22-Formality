slang=en
tlang=$1

data_dir=internal_split/${slang}-${tlang}
out_data_dir=processed_data/${slang}-${tlang}
mkdir -p $out_data_dir

for SET in train dev; do
    if [ $SET == "train" ]; then
      domains="combined"
    else
      domains="telephony topical-chat combined"
    fi

    for domain in $domains; do
      sacremoses -l en tokenize -x < ${data_dir}/$SET.${domain}.en > $out_data_dir/$SET.${domain}.en.tok
      subword-nmt apply-bpe --codes models/model.en-${tlang}/bpe_codes < $out_data_dir/$SET.${domain}.en.tok > $out_data_dir/$SET.${domain}.en.tok.bpe
      for direction in formal informal; do
        sacremoses -l ${tlang} tokenize -x < ${data_dir}/$SET.${domain}.${direction}.${tlang} > $out_data_dir/$SET.${domain}.${direction}.${tlang}.tok
        subword-nmt apply-bpe --codes models/model.en-${tlang}/bpe_codes < $out_data_dir/$SET.${domain}.${direction}.${tlang}.tok > $out_data_dir/$SET.${domain}.${direction}.${tlang}.tok.bpe
      done;
    done;
done
