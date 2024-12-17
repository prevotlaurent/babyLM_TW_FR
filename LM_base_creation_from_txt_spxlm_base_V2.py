import torch

from pathlib import Path
import tqdm as notebook_tqdm
import os
import re
import argparse
import json

import sentencepiece as spm

from tokenizers.processors import BertProcessing

from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from transformers import EarlyStoppingCallback, IntervalStrategy

from transformers import convert_slow_tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import XLMRobertaTokenizerFast, XLMRobertaTokenizer
from transformers import XLMRobertaConfig, XLMRobertaForMaskedLM
from transformers import AutoTokenizer

from datasets import load_dataset

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='')
parser.add_argument('-exp_tag', action="store", dest="exp_tag", default = '', type=str)
parser.add_argument('-language', action="store", dest="language", default = None, type=str)
parser.add_argument('-corpus_name', action="store", dest="corpus_name", default = None, type=str)
parser.add_argument('-tok_name', action="store", dest="tok_name", default = 'sp', type=str)
parser.add_argument('-tok_type', action="store", dest="tok_type", default = 'unigram', type=str)

parser.add_argument('-save_total_limit', action="store", dest="save_total_limit", default = 1, type=int)
parser.add_argument('-use_valid_data', action="store", dest="use_valid_data", default = True, type=boolean_string)

parser.add_argument('-epoch', action="store", dest="epoch", default = 50, type=int)
parser.add_argument('-batch_size', action="store", dest="batch_size", default = 32, type=int)
parser.add_argument('-vocab_size', action="store", dest="vocab_size", default = 10000, type=int)
parser.add_argument('-model_path', action="store", dest="model_path", default = "./models/", type=str)
parser.add_argument('-patience', action="store", dest="patience", default = 10, type=int)
parser.add_argument('-learning_rate', action="store", dest="learning_rate", default = 1e-4, type=float)
parser.add_argument('-group_texts', action="store", dest="group_texts", default = False, type=boolean_string)


arguments = parser.parse_args()

if arguments.exp_tag != '':
    p_exp_tag = '_' + arguments.exp_tag
else:
    p_exp_tag = ''
    
p_use_valid_data = arguments.use_valid_data
print(p_use_valid_data)

p_language = arguments.language
p_corpus_name = arguments.corpus_name
p_tok_name = arguments.tok_name
p_tok_type = arguments.tok_type
p_save_total_limit = arguments.save_total_limit
p_epoch = arguments.epoch
p_vocab_size = arguments.vocab_size
p_batch_size = arguments.batch_size
p_model_path = arguments.model_path
p_patience = arguments.patience
p_learning_rate = arguments.learning_rate
p_group_texts = arguments.group_texts

p_model_name =  p_language + '_'+ p_corpus_name + '_'+p_tok_name +  p_exp_tag
p_save_path = p_model_path + p_model_name

if not os.path.exists(p_save_path):
    os.makedirs(p_save_path)
    os.makedirs(p_save_path+'/sp/')
#    os.makedirs(p_save_path+'/converted/')
    
train_path = './data_raw_txt/'+p_language+'/'+p_corpus_name+'_train_sample.txt'
valid_path = './data_raw_txt/'+p_language+'/'+p_corpus_name+'_dev_sample.txt'

print(p_save_path)

#################################
#Train SentencePience Tokenizer #
#################################

spm.SentencePieceTrainer.train(
    input=train_path,
    model_prefix=p_model_name,
    vocab_size=p_vocab_size,
    model_type=p_tok_type,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=["<mask>"]
)

# SentencePiece trainer does not allow to specify an output folder...
os.rename(p_model_name+".model",p_save_path+'/sp/'+p_model_name+".model")
os.rename(p_model_name+".vocab",p_save_path+'/sp/'+p_model_name+".vocab")

sp = spm.SentencePieceProcessor(model_file=p_save_path+'/sp/'+p_model_name+'.model')
#vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}

#with open(p_save_path+"/sp/vocab.json", "w", encoding="utf-8") as vocab_file:
#    json.dump(vocab, vocab_file, ensure_ascii=False)
#with open(p_save_path+"/sp/merges.txt", "w") as merges_file:
#    merges_file.write("# No merges for SentencePiece\n")

sp.vocab_file = p_save_path+'/sp/'+p_model_name+'.model'
#sp_converter = convert_slow_tokenizer.SpmConverter(sp)
#converted = sp_converter.converted()
#converted.save(p_save_path+'/tokenizer.json')

tokenizer = XLMRobertaTokenizer(vocab_file=sp.vocab_file, max_len=512,clean_up_tokenization_spaces = False,
                                         return_special_tokens = True)

# tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=p_save_path,
                                              # clean_up_tokenization_spaces=False, 
                                              # pad_token='<pad>', 
                                              # unk_token='<unk>', 
                                              # bos_token='<s>', 
                                              # eos_token='</s>', 
                                              # mask_token='<mask>', 
                                              # model_max_length=512, 
                                              # padding_side='right', 
                                              # truncation_side='right')

tokenizer.save_pretrained(p_save_path)

#################################
######### XLM Roberta ###########
#################################

# Define model configuration
config = XLMRobertaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = XLMRobertaForMaskedLM(config)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

#tokenizer = XLMRobertaTokenizerFast.from_pretrained(p_save_path,max_len=512,clean_up_tokenization_spaces = False)

tokenizer = AutoTokenizer.from_pretrained(p_save_path,max_len=512,clean_up_tokenization_spaces = False)

# Dataset
if p_use_valid_data:
    dataset = load_dataset("text", data_files={"train": [train_path], "valid": [valid_path]})
else:
    dataset = load_dataset("text", data_files={"train": [train_path]})
    

# Create Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

def tokenize_function_concat(sample):
    # For concatenated version (no padding, no truncation)
    # because we're using concatenated texts now, so there is no need to pad now
    return tokenizer(
        sample['text'],
        padding=False,
        truncation=False,
        return_special_tokens_mask=True,
        return_token_type_ids = False,
    )

def tokenize_function_base(sample):
    return tokenizer(
        sample['text'],
        padding=False,
        truncation=True,
        return_special_tokens_mask=True,
        return_token_type_ids = False,
    )

def group_texts(examples, block_size=512):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop,
    # you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # result["labels"] = result["input_ids"].copy()
    return result


if p_group_texts :
    ###### CONCATENATED
    tokenized_train = dataset['train'].map(
        tokenize_function_concat,
        batched=True,
        remove_columns=['text'],
        load_from_cache_file=False,
    )
    if p_use_valid_data:
        tokenized_valid = dataset['valid'].map(
            tokenize_function_concat,
            batched=True,
            remove_columns=['text'],
            load_from_cache_file=False,
        )

    tokenized_train = tokenized_train.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
        load_from_cache_file=False,
    )
    
    if p_use_valid_data:
        tokenized_valid = tokenized_valid.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=4,
            load_from_cache_file=False,
        )



else:
    ###### NOT CONCATENATED
    # Create Dataset
    tokenized_train = dataset['train'].map(
        tokenize_function_base,
        batched=True,
        remove_columns=['text'],
        load_from_cache_file=False,
    )
    
    if p_use_valid_data:
        tokenized_valid = dataset['valid'].map(
            tokenize_function_base,
            batched=True,
            remove_columns=['text'],
            load_from_cache_file=False,
        )



if p_use_valid_data:

    training_args = TrainingArguments(
        output_dir=p_save_path,
        overwrite_output_dir=True,
        num_train_epochs=p_epoch,
        per_device_train_batch_size=p_batch_size,
        per_device_eval_batch_size=p_batch_size,  # evaluation batch size
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=p_save_total_limit,
        prediction_loss_only=True,
        save_only_model=True,
        learning_rate=p_learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=p_patience)]
    )

else:
    training_args = TrainingArguments(
        output_dir=p_save_path,
        overwrite_output_dir=True,
        num_train_epochs=p_epoch,
        per_device_train_batch_size=p_batch_size,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=p_save_total_limit,
        prediction_loss_only=True,
        save_only_model=True,
        learning_rate=p_learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train,
    )    


###############################
#    dataset = LineByLineTextDataset(
#        tokenizer=tokenizer,
#         file_path=train_path,
#         block_size=128,
#     )
#
#     # Define other args
#     training_args = TrainingArguments(
#         output_dir=p_save_path,
#         overwrite_output_dir=True,
#         num_train_epochs=p_epoch,
#         per_gpu_train_batch_size=64,
#         save_steps=10000,
#         save_total_limit=2,
#         prediction_loss_only=True,
#     )
#
#
#     # Define Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=dataset,
#     )
####################

trainer.save_model()
trainer.save_state()

trainer.train()
trainer.save_model()
trainer.save_state()

# Save Models
trainer.save_model(p_save_path)
