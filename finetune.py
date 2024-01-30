import evaluate
import os
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric, DatasetDict
import datasets
import random
from math import sqrt
import pandas as pd
import numpy as np
from IPython.display import display, HTML
from transformers import AutoTokenizer
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    MarianMTModel,
    MarianTokenizer,
)


os.environ["WANDB_DISABLED"] = "true"

SOURCE_LANG = "en"
TARGET_LANG = "it"

raw_datasets = load_dataset("Eugenememe/netflix-en-it")
metric = evaluate.load("sacrebleu")

# Tokenizer and model checkpoint
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

PREFIX = ""
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128


# Preprocessing function
def preprocess_function(examples):
    inputs = [PREFIX + ex[SOURCE_LANG] for ex in examples["translation"]]
    targets = [ex[TARGET_LANG] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


preprocess_function(raw_datasets["train"][:2])
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

BATCH_SIZE = 84
MODEL_NAME = MODEL_CHECKPOINT.rsplit("/", maxsplit=1)[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{MODEL_NAME}-finetuned",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    logging_dir="./logs",  # set this to a directory where you want to store logs
    logging_steps=100,  # log training information every 50 steps
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


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
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(f"{MODEL_NAME}-finetuned")

for dirname, _, filenames in os.walk(f"{MODEL_NAME}-finetuned"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


src_text = [
    "My name is Sarah, I live in London with my family and I absolutely love every bit of it."
]

FINETUNED_MODEL_NAME = f"{MODEL_NAME}-finetuned"
tokenizer = MarianTokenizer.from_pretrained(FINETUNED_MODEL_NAME)
finetuned_model = MarianMTModel.from_pretrained(FINETUNED_MODEL_NAME)
translated = finetuned_model.generate(
    **tokenizer(src_text, return_tensors="pt", padding=True)
)

print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
