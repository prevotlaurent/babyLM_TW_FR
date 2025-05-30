#!/usr/bin/env python
# coding: utf-8


import math
import pandas as pd
import numpy as np
import collections

import datasets
from datasets import Dataset, Value, ClassLabel, Features
from evaluate import load

import torch
from torch import nn

import transformers
#from transformers import RobertaTokenizerFast
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer

from transformers import EarlyStoppingCallback


#import seaborn as sns

import shutil

import os
import re

import argparse

import sentencepiece as spm

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='')


#parser.add_argument('-overall_name', action="store", dest="overall_name", default = '', type=str)
parser.add_argument('-task', action="store", dest="task", default = "red", type=str)
parser.add_argument('-lge', action="store", dest="lge", default = "fr", type=str)
parser.add_argument('-corpus', action="store", dest="corpus",default = 'buckeye',type=str)
parser.add_argument('-benchmark_file', action="store", dest="benchmark_file",default = '',type=str)
parser.add_argument('-benchmark_sep', action="store", dest="benchmark_sep",default = ',',type=str)
parser.add_argument('-label_column', action="store", dest="label_column",default = 'red',type=str)
parser.add_argument('-tok_column', action="store", dest="tok_column",default = 'tok',type=str)
parser.add_argument('-spk_column', action="store", dest="spk_column",default = 'speaker',type=str)

parser.add_argument('-exp_tag', action="store", dest="exp_tag", default = '', type=str)
parser.add_argument('-ckpt', action="store", dest="ckpt", default = "", type=str)

parser.add_argument('-run_only', action="store", dest="run_only", default = 0, type=int)
parser.add_argument('-compact', action="store", dest="compact", default = True, type=boolean_string)
parser.add_argument('-voc_prune', action="store", dest="voc_prune", default = True, type=boolean_string)

parser.add_argument('-ft_eps', action="store", dest="ft_eps", default = 10, type=int)
parser.add_argument('-patience', action="store", dest="patience", default = 3, type=int)

parser.add_argument('-prom_cutoff', action="store", dest="prom_cutoff", default = 1.25, type=float)
parser.add_argument('-red_cutoff', action="store", dest="red_cutoff", default = 0.75, type=float)


parser.add_argument('-batch_size', action="store", dest="batch_size", default = 32, type=int)
parser.add_argument('-max_length', action="store", dest="max_length", default = 128, type=int)
parser.add_argument('-lr', action="store", dest="lr", default = 2e-5, type=float)

parser.add_argument('-freeze_to', action="store", dest="freeze_to", default = 0.0, type=float)
parser.add_argument('-results_dir', action="store", dest="results_dir", default = "./results/", type=str)
parser.add_argument('-models_dir', action="store", dest="models_dir", default = "./models_ft/", type=str)
parser.add_argument('-figs_dir', action="store", dest="figs_dir", default = "./figs/", type=str)


args = parser.parse_args()

RESULT_FOLDER = args.results_dir
MODELS_FOLDER = args.models_dir
FIGS_FOLDER = args.figs_dir
LOGS_FOLDER = re.sub("/$", "_logs/", args.results_dir)




for FOLDER in [RESULT_FOLDER, FIGS_FOLDER, MODELS_FOLDER, LOGS_FOLDER]:
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
        
 

if args.exp_tag != '':
    exp_tag = '_' + args.exp_tag
else:
    exp_tag = ''
    

task = args.task
lge = args.lge
corpus = args.corpus
benchmark_file = args.benchmark_file
benchmark_sep = args.benchmark_sep
label_column = args.label_column
tok_column = args.tok_column

prom_cutoff = args.prom_cutoff
red_cutoff = args.red_cutoff

freeze_to = args.freeze_to

#overall_name = 'test'

run_only = args.run_only
compact = args.compact

checkpoint = args.ckpt
ft_eps = args.ft_eps
patience = args.patience
batch_size = args.batch_size
max_length = args.max_length
lr = args.lr
voc_prune = args.voc_prune


def normalize_tokens(row):
    tmp_tok = row[tok_column].lower()
    tmp_tok = tmp_tok.replace("'",'').replace('=','').replace('_','').replace('-','').replace('@@','*').replace('@','*').replace('#',",").replace('dummy',",")
    return tmp_tok

print("load tokenizer")
tokenizer = AutoTokenizer.from_pretrained(checkpoint,max_len=max_length,add_prefix_space=True)


if corpus == 'cid':
	FOLDS = {1:['AB','CM'],2:['YM','AG'],3:['EB','SR'],4:['LL','NH'],
		     5:['BX','MG'],6:['AP','LJ'],7:['IM','ML'],8:['MB','AC']}
elif corpus == 'buckeye':
	#FOLDS = {1:['s01', 's02', 's06', 's03'], 2:['s04', 's05', 's11', 's10'], 3:['s08', 's07', 's13', 's19'],
	#	 4:['s09', 's14', 's15', 's22'], 5:['s12', 's16', 's28', 's23'], 6:['s21', 's17', 's30', 's24'],
	#	 7:['s26', 's18', 's32', 's29'], 8:['s31', 's20', 's33', 's35'], 9:['s37', 's25', 's34', 's36'],
	#	 10:['s39', 's27', 's40', 's38']} 
	FOLDS = {1:['s01', 's02', 's06', 's03', 's37'], 2:['s04', 's05', 's11', 's10', 's25'], 
             3:['s08', 's07', 's13', 's19', 's34'], 4:['s09', 's14', 's15', 's22', 's36'],
             5:['s12', 's16', 's28', 's23', 's39'], 6:['s21', 's17', 's30', 's24', 's27'], 
             7:['s26', 's18', 's32', 's29', 's40'], 8:['s31', 's20', 's33', 's35', 's38']}         
elif corpus == 'mcdc':
	FOLDS = {1:['MCDC_01'], 2:['MCDC_02'], 3:['MCDC_03'],
		 4:['MCDC_05'], 5:['MCDC_09'], 6:['MCDC_10'],
		 7:['MCDC_25'], 8:['MCDC_26']}		 
         
NB_FOLDS = len(FOLDS.keys())    

print("load metric")
metric = load("./seqeval")

print("load read_csv")
df = pd.read_csv(benchmark_file, sep = benchmark_sep)

df[tok_column] = df[tok_column].fillna(',')
df[tok_column] = df.apply(normalize_tokens,axis=1)
#df['duration'] = df['end']-df['start']

if label_column == "prom":
    df['label'] = df[label_column]>prom_cutoff
elif label_column == "red":
    df['label'] = df[label_column]<red_cutoff
elif label_column == "bc":
    df['label'] = df[label_column]

def addfold(spk,folds):
    for fold in folds.keys():
        if spk in folds[fold]:
            return fold

print("load fold")
df['fold'] = df.apply(lambda row: addfold(row['speaker'], FOLDS), axis=1)


def small_data(df, keep, folds):
    folds[-1] = []
    targets = [x for x in folds.keys() if x > 0]
    for t in targets:
        folds[-1] = folds[-1] + folds[t][keep:] 
        folds[t] = folds[t][:keep]
        
    df['fold'] = df.apply(addfold,axis=1)
    df = df[df['fold'] > 0] 

    return df

if run_only > 0:
    df = small_data(df = df, keep = run_only, folds = FOLDS)
    
    
def token2sent(df,threshold=0.5):
    res = []
    
    tmp_toks = []
    tmp_labels = []

    for index,row in df.iterrows():
        if (row[tok_column] in ['#','dummy',',']) and row['duration'] > threshold:
            if tmp_toks != []:
                res.append([tmp_toks,tmp_labels,row['fold']])#
                tmp_toks = []
                tmp_labels = []
        else:
            tmp_toks.append(row[tok_column])
            tmp_labels.append(int(row['label']))
    output = pd.DataFrame(res,columns=[tok_column,'labels','fold'])
    output[tok_column] = output[tok_column].apply(lambda lst: [tokenizer.bos_token] + lst + [tokenizer.eos_token])
    output['labels'] = output["labels"].apply(lambda lst: [0] + lst + [0])
    return output

print("token2sent")
df_ready = token2sent(df)


def compact(df):

    res = []
    temp_toks = []
    temp_labels = []
    
    curr_fold = df.fold[0]
    
    for index,row in df.iterrows():
        if len(temp_toks) + len(row[tok_column]) <= max_length and row['fold'] == curr_fold:
            temp_toks = temp_toks + row[tok_column]
            temp_labels = temp_labels + row['labels']
        else:
            res.append([temp_toks,temp_labels,row['fold']])
            temp_toks = row[tok_column]
            temp_labels = row['labels']
        curr_fold = row['fold']

    return pd.DataFrame(res,columns=[tok_column,'labels','fold'])

print("compact")

if compact:
    df_ready = compact(df_ready)
    

## Create HF datasets from the dataframes
def bin2bio(row,label_str):
    binlist = row['labels']
    res = []
    prev = 0
    for item in binlist:
        if item == 1 :
            if prev != 1:
                res.append('B-'+label_str) 
            else:
                res.append('I-'+label_str)
            prev = 1
        else:
            res.append('O')
            prev = 0
       
    return res

df_ready['labels_bio'] = df_ready.apply(lambda row: bin2bio(row,task.upper()),axis=1)


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


print("dataset processing")
dataset = Dataset.from_pandas(df_ready)
dataset = dataset.map(lambda ex: {"tags": ex["labels_bio"]}) #red_bio
all_labels = get_label_list(dataset["tags"])
dataset = dataset.cast_column("tags", datasets.Sequence(datasets.ClassLabel(names=all_labels)))
label_list = dataset.features["tags"].feature.names

def tokenize_and_align_labels(examples,tokenizer):

    tokenized_inputs = tokenizer(examples[tok_column], padding='max_length', truncation=True, is_split_into_words=True, add_special_tokens = False, max_length = max_length)
    
    tokenized_inputs['input_tokens'] = []
    for i in range(len(tokenized_inputs.input_ids)):
        tokens = [x for x in tokenizer.convert_ids_to_tokens(tokenized_inputs.input_ids[i]) if x not in tokenizer.all_special_tokens]
        tokenized_inputs['input_tokens'].append(tokens)  

    labels = []
    
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        token_ids = tokenized_inputs.input_ids[i]
        previous_word_idx = None
        label_ids = []
        for word_idx in range(len(word_ids)):
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_ids[word_idx] is None or token_ids[word_idx] in tokenizer.all_special_ids:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_ids[word_idx] != previous_word_idx:
                label_ids.append(label[word_ids[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                # Modified to make sure it follows the BIO scheme
                if label[word_ids[word_idx]]==0:
                    good_label = 1
                else:
                    good_label = label[word_ids[word_idx]]
                label_ids.append(good_label)
            previous_word_idx = word_ids[word_idx]

        labels.append(label_ids)
 
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
    
def prepare_compute_metric_with_labellist(label_list):
    def compute_metric(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
    
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "f1": results["overall_f1"],
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "accuracy": results["overall_accuracy"],
        }
    return compute_metric
    
 

 
def run_one_fold(task,fold,split_dataset,checkpoint,tokenizer,label_list,weighted=False,verbose=True,error_analysis=False,keep_models=False):
    
    print(fold)
 
    tokenized_split_dataset = split_dataset.map(lambda d : tokenize_and_align_labels(d,tokenizer), batched=True)

    print("start run_complete_expe")

    model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=len(label_list), trust_remote_code=True)

    model_name = re.sub("(\./|/)", "_", checkpoint + exp_tag) +'-'+ str(fold)
        
    print(model_name)
             
    #layer freezing code goes here
    if freeze_to > 0.0:        

        for name, param in model.named_parameters():
            if 'embedding' in name:
                param.requires_grad = False
                print(name)
            elif 'embed_tokens' in name:
                param.requires_grad = False
                print(name)                
    
        freeze_to_layer = int(round(model.config.num_hidden_layers*freeze_to))
        freeze_to_layer   
        
        freeze_to_alpha = freeze_to_layer * 2
        
        for layer in range(0,freeze_to_layer):
            for name, param in model.named_parameters():
                if 'encoder.layer.' + str(layer) + '.' in name:
                    param.requires_grad = False
                    print(name)
                elif 'layers.' + str(layer) + '.' in name:
                    param.requires_grad = False
                    print(name)

        for alpha in range(0,freeze_to_alpha):
            for name, param in model.named_parameters():
                if 'alphas.' + str(alpha) in name:
                    param.requires_grad = False
                    print(name)
         
    if verbose:
        print(model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device in use:",device)
    
    #model.to('cuda')                                      #####
    

   
    args = TrainingArguments(
        MODELS_FOLDER+model_name+"-finetuned-"+task,
        evaluation_strategy = "epoch",
        #evaluation_strategy = "no",
        save_strategy ="epoch",
        logging_strategy="epoch",        
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=ft_eps,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model = 'f1',
        greater_is_better =True
        )

    
    data_collator = DataCollatorForTokenClassification(tokenizer)

    compute_metric = prepare_compute_metric_with_labellist(label_list)
    
    if weighted:
        trainer = CustomTrainer(model,args,
                          tokenized_split_dataset["train"],tokenized_split_dataset["valid"],
                          data_collator,tokenizer,compute_metrics=compute_metric,
                          callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
                          )

    else:
        trainer = Trainer(model,args,
                          train_dataset=tokenized_split_dataset["train"],
                          eval_dataset=tokenized_split_dataset["valid"],
                          data_collator=data_collator,tokenizer=tokenizer,compute_metrics=compute_metric,
                          callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
                          ) 
        
    print(args)
    trainer.train()
    trainer.save_state()
    
    predictions, labels, _ = trainer.predict(tokenized_split_dataset["test"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
        ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
        ]
    
    if error_analysis:
        EXP_FOLDER = LOGS_FOLDER + '/error_analysis_' + task+"_"+ re.sub("(\./|/)", "_", checkpoint + exp_tag) + '/'
        if not os.path.exists(EXP_FOLDER):
            os.makedirs(EXP_FOLDER)
        EA_df = pd.DataFrame(tokenized_split_dataset["test"])
        EA_df['predict'] = true_predictions
        EA_df['gold'] = true_labels
        #EA_df.to_csv(EXP_FOLDER+"error_analysis_"+task+"_"+model_name+'.csv')
        EA_df.to_csv(EXP_FOLDER+"complex_EA_"+str(fold)+'.csv')    
        
        tokens_col = [item for row in EA_df['input_tokens'] for item in row]
        predict_col = [item for row in EA_df['predict'] for item in row]
        gold_col = [item for row in EA_df['gold'] for item in row]
        EA_df = pd.DataFrame(list(zip(tokens_col, predict_col, gold_col)),
                       columns =['token', 'prediction', 'gold'])        
        EA_df.to_csv(EXP_FOLDER+'simple_EA_'+str(fold)+'.csv')
        

    if not keep_models:
        shutil.move(MODELS_FOLDER+model_name+"-finetuned-"+task+'/trainer_state.json', EXP_FOLDER+'trainer_state_'+str(fold)+'.json')
        shutil.rmtree(MODELS_FOLDER+model_name+"-finetuned-"+task)
    
    return metric.compute(predictions=true_predictions, references=true_labels)
    

def run_crossvalid(task,base_dataset,checkpoint,label_list,weighted=False,verbose=True,error_analysis=False,keep_models=False):
    
    #tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint, max_len=512,add_prefix_space=True)
    
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    results = {'fs':[],'prec':[],'rec':[]}
    for i in range(1,NB_FOLDS+1):
        print(i)
        split_ds = datasets.DatasetDict({
            'train': base_dataset.filter(lambda example: example["fold"] not in [i,(i%NB_FOLDS+1)]),
            'test': base_dataset.filter(lambda example: example["fold"] == i),
            'valid': base_dataset.filter(lambda example: example["fold"] == i%NB_FOLDS+1)
        })
        print(label_list)
        res = run_one_fold(task,i,split_ds,checkpoint,tokenizer,label_list,weighted,verbose,error_analysis,keep_models)
        print(res)
        results['fs'].append(res['overall_f1'])
        results['prec'].append(res['overall_precision'])
        results['rec'].append(res['overall_recall'])
    return results

def run_complete_expe(expe_name,ds,label_list,weighted=False):

    print('====')
    print(expe_name)
    print('====')    
       
    m_name = re.sub("(\./|/)", "_", checkpoint + exp_tag)
    print('running ' + str(m_name))
    res_cv = run_crossvalid(expe_name,ds,checkpoint,label_list,
                                weighted=weighted,verbose=False,error_analysis=True,keep_models=False)
    
    
    res_cv_df = pd.DataFrame(res_cv)
    res_cv_df['model'] = m_name
    res_cv_df.to_csv(RESULT_FOLDER+expe_name+'_'+m_name+'_cv.csv')

    
    return 0

print("start run_complete_expe")

run_complete_expe(args.lge+'_'+args.task,dataset,label_list)


