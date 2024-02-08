import os
import torch
import evaluate
import numpy as np
 
from transformers import (
    T5Tokenizer,
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import datasets
from datasets import load_dataset
from datasets import Dataset

import wandb
os.environ["WANDB_PROJECT"] = "T5-tuning"

MODEL = 'google/t5-v1_1-large'
# MODEL = 't5-large'
# MODEL = '/groups/3/gac50543/migrated_from_SFA_GPFS/share/kat5-base'
BATCH_SIZE = 64
NUM_PROCS = 16
EPOCHS = 15
OUT_DIR = 'results_t5-large-mt-1m'
MAX_LENGTH = 256 # Maximum context length to consider while preparing dataset.

src = "de"
trg = "en"

# load bleu for validation
bleu = evaluate.load("bleu")

model = T5ForConditionalGeneration.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# new tokens
new_tokens = ["<LOC>","</LOC>","<ORG>","</ORG>","<PERSON>","</PERSON>"]
# check if the tokens are already in the vocabulary
new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(list(new_tokens))
# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer))



# Amend the task-specific parameters
# task_params = model.config.task_specific_params
task_params = dict()
task_params['entity_translation_en_to_de'] = {'early_stopping': True, 'max_length': 300, 'num_beams': 4, 'prefix': 'entity translate English to German: '}
task_params['entity_translation_de_to_en'] = {'early_stopping': True, 'max_length': 300, 'num_beams': 4, 'prefix': 'entity translate German to English: '}
task_params['entity_recognition_en'] = {'early_stopping': True, 'max_length': 300, 'num_beams': 4, 'prefix': 'recognize English named entities: '}
task_params['entity_recognition_de'] = {'early_stopping': True, 'max_length': 300, 'num_beams': 4, 'prefix': 'recognize German named entities: '}

model.config.task_specific_params = task_params

# use_cahce is incompatible with gradient checkpointing
model.config.use_cache = False



def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in the preds as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # labels[labels == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # print(f"\nCalculating BLEU\n")
    bleu_output = bleu.compute(predictions=pred_str, references=label_str, max_order=4)
    with open(OUT_DIR+"/devel-predictions.txt", "w") as f:
        for sentence in pred_str:
            f.write(sentence + "\n")
    return {"bleu": round(np.mean(bleu_output["bleu"])*100, 2)}

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch[src], text_target=batch[trg], padding='max_length', truncation=True, max_length=MAX_LENGTH)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["labels"] = inputs.labels
  return batch

# Load data from files
data_dir = '/groups/3/gac50543/migrated_from_SFA_GPFS/matiss/data/wmt23/distill/extracted/for-mega/ner-mt/for-mt5';


# file_src = open(data_dir+'/1m-ent-mt-train.src', 'r')
# file_trg = open(data_dir+'/1m-ent-mt-train.trg', 'r')
# lines_src = file_src.readlines()
# lines_trg = file_trg.readlines()

# file_data = {src : lines_src, trg : lines_trg}
# train_data = Dataset.from_dict(file_data)

# train_data = train_data.map(
#     process_data_to_model_inputs,
#     batched=True,
#     batch_size=512,
#     remove_columns=[src, trg],
#     num_proc=NUM_PROCS, # set to the number of CPU cores in AF node
# )

# train_data.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "labels"],
# )


data_path = "/groups/3/gac50543/migrated_from_SFA_GPFS/matiss/data/wmt23/distill/extracted/for-mega/ner-mt/for-mt5/mt5-large-data-save-1m"
# train_data.save_to_disk(data_path)
train_data = datasets.load_from_disk(data_path)


dfile_src = open(data_dir+'/sm-ent-mt-devel.up.src', 'r')
dfile_trg = open(data_dir+'/sm-ent-mt-devel.up.trg', 'r')
dlines_src = dfile_src.readlines()
dlines_trg = dfile_trg.readlines()

dfile_data = {src : dlines_src, trg : dlines_trg}
val_data = Dataset.from_dict(dfile_data)
val_data = val_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=512,
    remove_columns=[src, trg],
    num_proc=NUM_PROCS,
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")


training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    logging_dir=OUT_DIR,
    evaluation_strategy='steps',
    logging_steps=250,
    eval_steps=1000,
    save_steps=1000,
    warmup_steps=5000,
    save_total_limit=3,
    learning_rate=0.0001,
    weight_decay=0.01,
    generation_max_length=128,
    fp16=True,
    load_best_model_at_end=True,
    predict_with_generate=True,
    auto_find_batch_size=True,
    # metric_for_best_model = 'bleu',
    metric_for_best_model = 'eval_loss',
    report_to="wandb",
    run_name="1m-"+MODEL+"-"+src+"-"+trg,
    seed=347155,
    # optim="adafactor",
    optim="adamw_torch",
    deepspeed="ds-config.json",
)
 
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)],
)

history = trainer.train()


# For training continuation from previous checkpoint

# resume_model_path = "/home/aad13940yw/tools/flant5/results_t5-small_11_mt_100k/second-save"
# history = trainer.train(resume_from_checkpoint=resume_model_path)

