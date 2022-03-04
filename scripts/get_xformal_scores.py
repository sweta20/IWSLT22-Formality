from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import sacrebleu
import sys
import os
import argparse
from utils import get_data, read_file
import torch
from tqdm import tqdm

CACHE_DIR="/fs/clip-scratch/sweagraw/CACHE"

def main():

	arg_parser = argparse.ArgumentParser(description='Extract the probability scores of being informal from trained classifiers')
	arg_parser.add_argument('--input', '-i', type=str, default=None)
	arg_parser.add_argument('--output', '-o', type=str, default=None)
	arg_parser.add_argument('--model-dir', '-m', type=str, default="/fs/clip-controllablemt/IWSLT2022/models/xformal-classifier")
	arg_parser.add_argument('--is-regression', dest='is_regression', action='store_true')

	args = arg_parser.parse_args()
	source = read_file(args.input)

	if args.is_regression:
		num_labels=1
		sub_dir="regression"
	else:
		num_labels=2
		sub_dir="classifier"

	config = AutoConfig.from_pretrained(
		args.model_dir,
		num_labels=num_labels,
		cache_dir=CACHE_DIR)
	tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
	model = AutoModelForSequenceClassification.from_pretrained(
		args.model_dir,
		from_tf=bool(".ckpt" in args.model_dir),
		config=config,
		cache_dir=CACHE_DIR)

	with open(args.output, "w") as f:
		scores = []
		for text in tqdm(source):
			model_inputs = tokenizer(text, return_tensors="pt",truncation=True, padding=True, max_length=200)
			logits=model(**model_inputs).logits
			if args.is_regression:
				f.write(str(logits.tolist()[0][0])+ "\n")
			else:
				# only write scores for informal class
				f.write(str(torch.softmax(logits, dim=1).tolist()[0][0])+ "\n")
	

if __name__ == '__main__':
	main()