
slang=en

for tlang in hi ja es de; do
	output_data_dir=internal_split/${slang}-${tlang}
	mkdir -p ${output_data_dir}
	for domain in topical-chat telephony; do
		for direction in formal informal; do
			data_dir=data/train/${slang}-${tlang}
			# create train/dev
			if [ $tlang == "ja" ]; then
			  head -n450 ${data_dir}/formality-control.train.${domain}.en-${tlang}.en > ${output_data_dir}/train.${domain}.en
			  head -n450 ${data_dir}/formality-control.train.${domain}.en-${tlang}.${direction}.${tlang} > ${output_data_dir}/train.${domain}.${direction}.${tlang}
			  head -n450 ${data_dir}/formality-control.train.${domain}.en-${tlang}.${direction}.annotated.${tlang} > ${output_data_dir}/train.${domain}.${direction}.annotated.${tlang}

			else
			  head -n150 ${data_dir}/formality-control.train.${domain}.en-${tlang}.en > ${output_data_dir}/train.${domain}.en
			  head -n150 ${data_dir}/formality-control.train.${domain}.en-${tlang}.${direction}.${tlang} > ${output_data_dir}/train.${domain}.${direction}.${tlang}
			  head -n150 ${data_dir}/formality-control.train.${domain}.en-${tlang}.${direction}.annotated.${tlang} > ${output_data_dir}/train.${domain}.${direction}.annotated.${tlang}
			fi;

			tail -n50 ${data_dir}/formality-control.train.${domain}.en-${tlang}.en > ${output_data_dir}/dev.${domain}.en
			tail -n50 ${data_dir}/formality-control.train.${domain}.en-${tlang}.${direction}.${tlang} > ${output_data_dir}/dev.${domain}.${direction}.${tlang}
			tail -n50 ${data_dir}/formality-control.train.${domain}.en-${tlang}.${direction}.annotated.${tlang} > ${output_data_dir}/dev.${domain}.${direction}.annotated.${tlang}

		done;
	done
done;


for tlang in hi ja es de; do
	output_data_dir=internal_split/${slang}-${tlang}
	for direction in formal informal; do
		cat ${output_data_dir}/train.topical-chat.en ${output_data_dir}/train.telephony.en > ${output_data_dir}/train.combined.en
		cat ${output_data_dir}/train.topical-chat.${direction}.${tlang} ${output_data_dir}/train.telephony.${direction}.${tlang} > ${output_data_dir}/train.combined.${direction}.${tlang}
		cat ${output_data_dir}/train.topical-chat.${direction}.annotated.${tlang} ${output_data_dir}/train.telephony.${direction}.annotated.${tlang} > ${output_data_dir}/train.combined.${direction}.annotated.${tlang}

		cat ${output_data_dir}/dev.topical-chat.en ${output_data_dir}/dev.telephony.en > ${output_data_dir}/dev.combined.en
		cat ${output_data_dir}/dev.topical-chat.${direction}.${tlang} ${output_data_dir}/dev.telephony.${direction}.${tlang} > ${output_data_dir}/dev.combined.${direction}.${tlang}
		cat ${output_data_dir}/dev.topical-chat.${direction}.annotated.${tlang} ${output_data_dir}/dev.telephony.${direction}.annotated.${tlang} > ${output_data_dir}/dev.combined.${direction}.annotated.${tlang}
	 done;
done;