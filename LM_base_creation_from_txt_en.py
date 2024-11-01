import torch

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer


from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from transformers import EarlyStoppingCallback, IntervalStrategy

import tqdm as notebook_tqdm
import os
import re
import argparse

from datasets import load_dataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('-epoch', action="store", dest="epoch", default = 100, type=int)
parser.add_argument('-batch_size', action="store", dest="batch_size", default = 32, type=int)
parser.add_argument('-vocab_size', action="store", dest="vocab_size", default = 10000, type=int)

parser.add_argument('-model_name', action="store", dest="model_name", default = "blah", type=str)
parser.add_argument('-model_path', action="store", dest="model_path", default = "./models/", type=str)
parser.add_argument('-patience', action="store", dest="patience", default = 10, type=int)
parser.add_argument('-learning_rate', action="store", dest="learning_rate", default = 1e-4, type=float)
parser.add_argument('-group_texts', action="store", dest="group_texts", default = True, type=bool)


arguments = parser.parse_args()

p_epoch = arguments.epoch
p_vocab_size = arguments.vocab_size
p_batch_size = arguments.batch_size
p_model_path = arguments.model_path
p_model_name = arguments.model_name
p_save_path = p_model_path + p_model_name + '_' + str(p_epoch)
p_patience = arguments.patience
p_learning_rate = arguments.learning_rate
p_alibi = arguments.alibi
p_group_texts = arguments.group_texts

if 'babylm' in p_model_name:
    train_path = './data_raw_txt/en/babylm_train_sample.txt'
    valid_path = './data_raw_txt/en/babylm_dev_sample.txt'
                
elif 'spoken' in p_model_name: 
    train_path = './data_raw_txt/en/spoken_train_sample.txt'
    valid_path = './data_raw_txt/en/spoken_dev_sample.txt'

elif 'written' in p_model_name: 
    train_path = './data_raw_txt/en/wiki_train_sample.txt'
    valid_path = './data_raw_txt/en/wiki_dev_sample.txt'

else:
    print('unspecified method!')
    exit()
    
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=[train_path], vocab_size=p_vocab_size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

if not os.path.exists(p_save_path):
    os.makedirs(p_save_path)
tokenizer.save_model(p_save_path)

tokenizer = ByteLevelBPETokenizer(
    p_save_path+"/vocab.json",
    p_save_path+"/merges.txt",
)

config = RobertaConfig(
    vocab_size=p_vocab_size,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#device = torch.device('cpu')

model.to(device)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

tokenizer = RobertaTokenizerFast.from_pretrained(p_save_path, max_len=512, clean_up_tokenization_spaces = False)

dataset = load_dataset("text", data_files={"train": [train_path], "valid": [valid_path]})

#because we're using concatenated texts now, so there is no need to pad now
def tokenize_function(sample):
    return tokenizer(
        sample['text'],
        #padding='max_length',
        padding=False,
        #truncation=True,
        truncation=False,
        return_special_tokens_mask=True,
        return_token_type_ids = False,
    )
    
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

tokenized_train = dataset['train'].map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],
    load_from_cache_file=False,
)
tokenized_valid = dataset['valid'].map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],
    load_from_cache_file=False,
)


def group_texts(examples, block_size = 512):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    #result["labels"] = result["input_ids"].copy()
    
    return result
    
if p_group_texts:
    
    tokenized_train = tokenized_train.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
        load_from_cache_file=False,
    )

    tokenized_valid = tokenized_valid.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
        load_from_cache_file=False,
    )

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=p_save_path,
    overwrite_output_dir=True,
    num_train_epochs=p_epoch,
    per_device_train_batch_size=p_batch_size,
    per_device_eval_batch_size=p_batch_size,  # evaluation batch size
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy ="epoch",
    logging_strategy="epoch",           
    save_total_limit=1,
    prediction_loss_only=True,
    save_only_model = True,
    learning_rate = p_learning_rate,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=p_patience)]

)

num_params = model.num_parameters()
outfile = open(p_save_path + '/params.txt', 'w')
outfile.write('model.num_parameters()' + str(num_params))
outfile.close()

trainer.save_model()
trainer.save_state()

trainer.train()
trainer.save_model()
trainer.save_state()

