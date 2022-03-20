from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import sacrebleu
import sys
import os
import argparse
from utils import get_data, read_file
sys.path.append("/fs/clip-controllablemt/IWSLT2022/notebooks/")
from mbart_covariate import CMBartForConditionalGeneration
import torch
import tqdm as tqdm

tgt_lang_to_code = {
	"hi" : "hi_IN",
	"de" : "de_DE",
	"es" : "es_XX",
	"it" : "it_IT",
	"ru" : "ru_RU",
	"ja" : "ja_XX"
}

direction_to_id = {
	"neutral":0,
	"formal":1,
	"informal":2
}

def translate_text(text, tgt_lang, model, tokenizer,  covariate_index=None, strategy="greedy"):
	model_inputs = tokenizer(text, return_tensors="pt", padding=True)
	kwargs = {}
	if covariate_index is not None:
		kwargs["covariate_ids"] = torch.tensor([covariate_index]*len(text))

	if torch.cuda.is_available():
		model = model.to("cuda:0")
		model_inputs = model_inputs.to("cuda:0")
		kwargs["covariate_ids"] = kwargs["covariate_ids"].to("cuda:0")

	if strategy == "greedy":
		generated_tokens = model.generate(
			**model_inputs,
			forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_to_code[tgt_lang]],
			**kwargs
		)
	else:
		generated_tokens = model.generate(
			**model_inputs,
			forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_to_code[tgt_lang]],
			max_length=50, 
			num_beams=5, 
			num_return_sequences=5, 
			early_stopping=True,
			**kwargs
		)
	return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

def main():

	arg_parser = argparse.ArgumentParser(description='Get mBART translations from trained model')
	arg_parser.add_argument('--domain', '-d', type=str, default=None)
	arg_parser.add_argument('--split', '-s', type=str, default=None)
	arg_parser.add_argument('--lang', '-l', type=str, default=None)
	arg_parser.add_argument('--exp-name', '-e', type=str, default="mBART")
	arg_parser.add_argument('--model-dir', '-m', type=str, default="facebook/mbart-large-50-one-to-many-mmt")
	arg_parser.add_argument('--is-covariate', dest='is_covariate', action='store_true')
	arg_parser.add_argument('--eval-direction', '-f', type=str, default=None)
	arg_parser.add_argument('--data-dir', type=str, default="internal_split")
	arg_parser.add_argument('--source', type=str, default=None)
	arg_parser.add_argument('--output', type=str, default=None)


	args = arg_parser.parse_args()
	domain=args.domain
	split=args.split
	src_lang="en"
	tgt_lang=args.lang

	if args.is_covariate:
		# Additive intervention added to encoder
		model = CMBartForConditionalGeneration.from_pretrained(args.model_dir, cache_dir="/fs/clip-scratch/sweagraw/CACHE")
		covariate_index = direction_to_id[args.eval_direction]
	else:
		model = MBartForConditionalGeneration.from_pretrained(args.model_dir, cache_dir="/fs/clip-scratch/sweagraw/CACHE")
		covariate_index = None
	tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX", tgt_lang=tgt_lang_to_code[tgt_lang], cache_dir="/fs/clip-scratch/sweagraw/CACHE")

	if args.source:
		source = read_file(args.source)
	else:
		source = read_file(f"{args.data_dir}/en-{tgt_lang}/{split}.{domain}.en")

	if args.output is None:
		output_dir=f"experiments/{src_lang}-{tgt_lang}/{args.exp_name}/{domain}"
		os.makedirs(output_dir, exist_ok=True)
		output_file = output_dir+"/out."+split
	else:
		output_file = args.output

	with open(output_file, "w") as f:
		for src in tqdm.tqdm(source):
			f.write(translate_text([src], tgt_lang, model, tokenizer, covariate_index)[0] + "\n")

if __name__ == '__main__':
	main()