from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import sacrebleu
import sys
import os
import argparse
from utils import get_data, read_file
import torch

CACHE_DIR="/fs/clip-scratch/sweagraw/CACHE"

def main():

	arg_parser = argparse.ArgumentParser(description='Extract scores from trained xformal model')
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

	scores = []
	for text in source:
		model_inputs = tokenizer(text, return_tensors="pt")
		logits=model(**model_inputs).logits
		if args.is_regression:
			scores.append(logits.tolist()[0][0])
		else:
			# only write scores for informal class
			scores.append(torch.softmax(logits, dim=1).tolist()[0][0])

	with open(args.output, "w") as f:
		for out in scores:
			f.write(str(out) + "\n")

	

if __name__ == '__main__':
	main()