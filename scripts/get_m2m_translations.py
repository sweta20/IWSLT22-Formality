from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer
import sacrebleu
import sys
import os
import argparse
from utils import get_data, read_file
import torch

def translate_text(text, tgt_lang, model, tokenizer,  covariate_index=None, strategy="greedy"):
	model_inputs = tokenizer(text, return_tensors="pt", padding=True)
	if strategy == "greedy":
		generated_tokens = model.generate(
			**model_inputs,
			forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
		)
	else:
		generated_tokens = model.generate(
			**model_inputs,
			forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
			max_length=50, 
			num_beams=5, 
			num_return_sequences=5, 
			early_stopping=True
		)
	return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

def main():

	arg_parser = argparse.ArgumentParser(description='Get mBART translations from trained model')
	arg_parser.add_argument('--domain', '-d', type=str, default=None)
	arg_parser.add_argument('--split', '-s', type=str, default=None)
	arg_parser.add_argument('--lang', '-l', type=str, default=None)
	arg_parser.add_argument('--exp-name', '-e', type=str, default="m2m100_418M")
	arg_parser.add_argument('--model-dir', '-m', type=str, default="facebook/m2m100_418M")
	arg_parser.add_argument('--is-covariate', dest='is_covariate', action='store_true')
	arg_parser.add_argument('--eval-direction', '-f', type=str, default=None)

	args = arg_parser.parse_args()
	domain=args.domain
	split=args.split
	src_lang="en"
	tgt_lang=args.lang

	if args.is_covariate:
		print("Not Implemented!")
	else:
		model = M2M100ForConditionalGeneration.from_pretrained(args.model_dir, cache_dir="/fs/clip-scratch/sweagraw/CACHE")
		covariate_index = None
	tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="en", cache_dir="/fs/clip-scratch/sweagraw/CACHE")

	source = read_file(f"internal_split/en-{tgt_lang}/{split}.{domain}.en")

	output_dir=f"experiments/{src_lang}-{tgt_lang}/{args.exp_name}/{domain}"
	os.makedirs(output_dir, exist_ok=True)

	# outputs = translate_text(source, tgt_lang, model, tokenizer, covariate_index)

	with open(output_dir+"/out."+split, "w") as f:
		for src in source:
			f.write(translate_text([src], tgt_lang, model, tokenizer, covariate_index)[0] + "\n")


if __name__ == '__main__':
	main()