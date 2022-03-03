import torch
import numpy as np
import random
import argparse
from transformers import AutoTokenizer
import sacrebleu
import sys
import os
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import load_dataset, load_metric
from utils import get_data, read_file
import torch
from torch.utils.data import Dataset
import numpy as np

sys.path.append("/fs/clip-controllablemt/IWSLT2022/notebooks/")
from mbart_covariate import MBartSeq2SeqTrainer

seed=1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
STEPS=100
MAX_LENGTH=64
tgt_lang_to_code = {
	"hi" : "hi_IN",
	"de" : "de_DE",
	"es" : "es_XX",
	"it" : "it_IT",
	"ru" : "ru_RU",
	"ja" : "ja_XX"
}

MODEL_NAME="facebook/mbart-large-50-one-to-many-mmt"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/fs/clip-scratch/sweagraw/CACHE")
metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
	preds = [pred.strip() for pred in preds]
	labels = [[label.strip()] for label in labels]

	return preds, labels

def compute_metrics(eval_preds):
	preds, labels = eval_preds
	if isinstance(preds, tuple):
		preds = preds[0]
	decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

	# Replace -100 in the labels as we can't decode them.
	labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

	# Some simple post-processing
	decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

	result = metric.compute(predictions=decoded_preds, references=decoded_labels)
	result = {"bleu": result["score"]}

	prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
	result["gen_len"] = np.mean(prediction_lens)
	result = {k: round(v, 4) for k, v in result.items()}
	return result


class FormalityData(Dataset):
	
	def __init__(self, domain, split, src_lang, tgt_lang, direction):
		self.source, self.formal_translations, self.informal_translations=get_data(tgt_lang, domain, split)
		tokenizer.src_lang = "en-XX"
		tokenizer.tgt_lang  = tgt_lang_to_code[tgt_lang]
		self.direction = direction
		self.max_target_length=MAX_LENGTH
		self.max_input_length=MAX_LENGTH
		self.model_inputs = self.encode_split()
		self.tgt_lang = tgt_lang
		
		
	def __len__(self):
		return len(self.model_inputs["input_ids"])
	
	def encode_split(self):
		model_inputs = tokenizer(self.source, max_length=self.max_input_length, truncation=True)
		with tokenizer.as_target_tokenizer():
			if self.direction == "formal":
				labels = tokenizer(self.formal_translations, max_length=self.max_target_length, truncation=True)
			else:
				labels = tokenizer(self.informal_translations, max_length=self.max_target_length, truncation=True)
		model_inputs["labels"] = labels["input_ids"]
		return model_inputs

	def __getitem__(self, idx):
		item = {k: v[idx] for k, v in self.model_inputs.items()}
		item["labels"] = self.model_inputs["labels"][idx]
		item["forced_bos_token_id"] = tokenizer.lang_code_to_id[tgt_lang_to_code[self.tgt_lang]]
		return item


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
	arg_parser = argparse.ArgumentParser(description='Finetune mBART model')
	arg_parser.add_argument('--domain', '-d', type=str, default="combined")
	arg_parser.add_argument('--direction', '-f', type=str, default=None)
	arg_parser.add_argument("--num-layers", default=2, type=int, help="Number of decoder layers to unfreeze")
	arg_parser.add_argument("--train-batch-size", default=8, type=int, help="Traning batch size")
	arg_parser.add_argument("--eval-batch-size", default=4, type=int, help="Eval Batch size")
	arg_parser.add_argument("--num-epochs", default=10, type=int, help="Number of epochs to train")
	arg_parser.add_argument("--learning-rate", default=3e-5, type=float, help="The amount of style attributes to be masked.")
	arg_parser.add_argument('--lang', type=str, default="xx")

	args = arg_parser.parse_args()
	model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir="/fs/clip-scratch/sweagraw/CACHE")

	model.requires_grad_(False)
	print(model.get_decoder().layers[-args.num_layers:])
	model.get_decoder().layers[-args.num_layers:].requires_grad_(True)
	model.lm_head.requires_grad_(True)
	print(f"Training {count_params(model)} parameters...")

	src_lang="en"
	train_datasets = []

	if args.lang == "xx":
		tgt_langs = ["de", "hi", "ja", "es"]
	else:
		tgt_langs = [args.lang]

	for tgt_lang in tgt_langs:
		train_datasets.append(FormalityData(args.domain, "train", src_lang, tgt_lang, args.direction))
	train_dataset = torch.utils.data.ConcatDataset(train_datasets)

	dev_datasets = []
	for tgt_lang in tgt_langs:
		dev_datasets.append(FormalityData(args.domain, "dev", src_lang, tgt_lang, args.direction))
	dev_dataset = torch.utils.data.ConcatDataset(dev_datasets)

	data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

	training_args = Seq2SeqTrainingArguments(
		output_dir=f"models/{MODEL_NAME}-finetuned-{src_lang}-to-{args.lang}-{args.direction}-test",
		evaluation_strategy="epoch",
		learning_rate=args.learning_rate,
		per_device_train_batch_size=args.train_batch_size,
		per_device_eval_batch_size=args.eval_batch_size,
		weight_decay=0.01,
		save_total_limit=10,
		save_strategy="epoch",
		logging_steps=STEPS,
		eval_steps=STEPS,
		num_train_epochs=args.num_epochs,
		gradient_accumulation_steps=2,
		predict_with_generate=True,
		fp16=True,
		push_to_hub=False,
	)

	trainer = MBartSeq2SeqTrainer(
		model,
		training_args,
		train_dataset=train_dataset,
		eval_dataset=dev_dataset,
		data_collator=data_collator,
		tokenizer=tokenizer,
		compute_metrics=compute_metrics
	)
	torch.cuda.empty_cache()
	trainer.train()

	trainer.save_model(f"models/{MODEL_NAME}-finetuned-{src_lang}-to-{args.lang}-{args.direction}-test")

if __name__ == '__main__':
	main()

