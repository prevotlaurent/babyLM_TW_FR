#!/usr/bin/env python
# coding: utf-8


import math
import pandas as pd
import numpy as np
import collections

import datasets
from datasets import Dataset, Value, ClassLabel, Features
from datasets import load_metric

import torch
from torch import nn

import transformers
from transformers import RobertaTokenizerFast
from transformers import AutoTokenizer

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer

from transformers import EarlyStoppingCallback



import seaborn as sns

import shutil

import os
import re

import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('-run_only', action="store", dest="run_only", default = 0, type=int)
parser.add_argument('-ft_eps', action="store", dest="ft_eps", default = 10, type=int)
parser.add_argument('-batch_size', action="store", dest="batch_size", default = 16, type=int)
parser.add_argument('-ckpt', action="store", dest="ckpt", default = "", type=str)
parser.add_argument('-task', action="store", dest="task", default = "red", type=str)
parser.add_argument('-max_length', action="store", dest="max_length", default = 128, type=int)
parser.add_argument('-overall_name', action="store", dest="", default = 'blah', type=str)
parser.add_argument('-lr', action="store", dest="lr", default = 2e-5, type=float)
parser.add_argument('-compact', action="store", dest="compact", default = True, type=bool)

RESULT_FOLDER ="results/"


args = parser.parse_args()

run_only = args.run_only
ft_eps = args.ft_eps
batch_size = args.batch_size
max_length = args.max_length
lr = args.lr

def normalize_tokens(row):
    tmp_tok = row['word'].lower()
    tmp_tok = tmp_tok.replace("'",'').replace('=','').replace('_','').replace('-','').replace('@@','*').replace('@','*').replace('#',",").replace('dummy',",")
    return tmp_tok


BENCHMARK_EN = './data_benchmark/buckeye_with_prom.tsv'

FIGS_FOLDER ="./figs/"
MODELS_FOLDER ="./models/"

checkpoint = args.ckpt
tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint, max_len=max_length,add_prefix_space=True)

overall_name = args.overall_name

for FOLDER in [RESULT_FOLDER, FIGS_FOLDER, MODELS_FOLDER]:
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)



df = pd.read_csv(BENCHMARK_EN, sep = '\t')



df['word'] = df['word'].fillna(',') 
df['word'] = df.apply(normalize_tokens,axis=1)



df['reduc_ratio'] = df['word_dur']/df['baseline_with_phoneme']
df['reduc_label'] = df['reduc_ratio']<0.7
df['prom_label'] = df['prom']>1.25


FOLDS = {
    1:['s01', 's02', 's06', 's03'],
2:['s04', 's05', 's11', 's10'],
3:['s08', 's07', 's13', 's19'],
4:['s09', 's14', 's15', 's22'],
5:['s12', 's16', 's28', 's23'],
6:['s21', 's17', 's30', 's24'],
7:['s26', 's18', 's32', 's29'],
8:['s31', 's20', 's33', 's35'],
9:['s37', 's25', 's34', 's36'],
10:['s39', 's27', 's40', 's38']}



def addfold(row):
    for fold in FOLDS.keys():
        if row['speaker'] in FOLDS[fold]:
            return fold

df['fold'] = df.apply(addfold,axis=1)


def small_data(df, keep = 1, folds = FOLDS):
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



## Create BIO Labels from binary
## There is an option to keep the labels binary (all '1' become 'B-XXX')

#from tokenper-row to utterance-per-row
def token2sent(df):
    res = []
    tmp_toks = []
    tmp_reds = []
    tmp_prom = []
    for index,row in df.iterrows():
#        print(row)
        if (row['word'] in ['#','dummy',',']):
            # newsent
#            print('newsent')
            if tmp_toks != []:
                res.append([tmp_toks,tmp_reds,tmp_prom,row['fold']])
                tmp_toks = []
                tmp_reds = []
                tmp_prom = []
                tmp_smths = []
        else:
#            print('addtok')
            tmp_toks.append(row['word'])
            tmp_reds.append(int(row['reduc_label']))
            tmp_prom.append(int(row['prom_label']))
            #tmp_smths.append(int(row['label_smooth_reduc']))
#            input("...")
    output = pd.DataFrame(res,columns=['word','reduc','prom','fold'])
    output["word"] = output["word"].apply(lambda lst: [tokenizer.bos_token] + lst + [tokenizer.eos_token])
    output['reduc'] = output["reduc"].apply(lambda lst: [0] + lst + [0])
    output['prom'] = output["prom"].apply(lambda lst: [0] + lst + [0])

    return output


df_ready = token2sent(df)



def compact(df):

    res = []
    temp_toks = []
    temp_reds = []
    temp_prom = []
    
    curr_fold = df.fold[0]
    
    for index,row in df.iterrows():
        #print(row['word'])
        #print(curr_fold)
        #print(row['fold'])
        #print(temp_toks)
        if len(temp_toks) + len(row['word']) <= max_length and row['fold'] == curr_fold:
            temp_toks = temp_toks + row['word']
            temp_reds = temp_reds + row['reduc']
            temp_prom = temp_prom + row['prom']
        else:
            res.append([temp_toks,temp_reds,temp_prom,row['fold']])
            temp_toks = row['word']
            temp_reds = row['reduc']
            temp_prom = row['prom']
        curr_fold = row['fold']

    return pd.DataFrame(res,columns=['word','reduc','prom','fold'])


if args.compact:
    df_ready = compact(df_ready)





## Create HF datasets from the dataframes
def bin2bio(row,row_name,label_str,pure_binary=False):
    binlist = row[row_name]
    res = []
    prev = 0
    for item in binlist:
        if item == 1 :
            if prev != 1:
                res.append('B-'+label_str) 
            else:
                if not pure_binary:
                    res.append('I-'+label_str)
                else:
                    res.append('B-'+label_str)
            prev = 1
        else:
            res.append('O')
            prev = 0
       
    return res



df_red_ready = df_ready[['word', 'reduc', 'fold']]
df_prom_ready = df_ready[['word', 'prom', 'fold']]

df_red_ready.loc[:,'label_bio'] = df_red_ready.apply(bin2bio,args=('reduc','RED',False),axis=1)
df_prom_ready.loc[:,'label_bio'] = df_prom_ready.apply(bin2bio,args=('prom','PROM',False),axis=1)



def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list



dataset_prom = Dataset.from_pandas(df_prom_ready)
dataset_prom = dataset_prom.map(lambda ex: {"tags": ex["label_bio"]}) #red_bio
all_labels_prom = get_label_list(dataset_prom["tags"])
dataset_prom = dataset_prom.cast_column("tags", datasets.Sequence(datasets.ClassLabel(names=all_labels_prom)))
label_list_prom = dataset_prom.features["tags"].feature.names


dataset_reduc = Dataset.from_pandas(df_red_ready)
dataset_reduc = dataset_reduc.map(lambda ex: {"tags": ex["label_bio"]}) #red_bio
all_labels_reduc = get_label_list(dataset_reduc["tags"])
dataset_reduc = dataset_reduc.cast_column("tags", datasets.Sequence(datasets.ClassLabel(names=all_labels_reduc)))
label_list_reduc = dataset_reduc.features["tags"].feature.names


## Function to handle labels and subwords
# adapted from HF examples
## Function to handle labels and subwords
# adapted from HF examples
def tokenize_and_align_labels(examples,tokenizer):
    
    tokenized_inputs = tokenizer(examples["word"], padding='max_length', truncation=True, is_split_into_words=True, add_special_tokens = False, max_length = max_length)        
    #print(tokenized_inputs.word_ids(batch_index=0))
    #exit()
    labels = []
    
    for i, label in enumerate(examples["tags"]):
        #print(i, label)
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





## Custom Trainer when we want weights for evaluating the labels
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 0.2]).to('cuda'))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss



metric = load_metric("seqeval")

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
 
    model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=3, trust_remote_code=True)
        
 
        
    if verbose:
        print(model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device in use:",device)
    
    #model.to('cuda')                                      #####
    
    model_name = re.sub("(\./|/)", "_", checkpoint) +'-'+ str(fold)
        
    print(model_name)
   
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
                          callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
                          )

    else:
        trainer = Trainer(model,args,
                          train_dataset=tokenized_split_dataset["train"],
                          eval_dataset=tokenized_split_dataset["valid"],
                          data_collator=data_collator,tokenizer=tokenizer,compute_metrics=compute_metric,
                          callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
                          ) 
        

    trainer.train()
    
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
        EA_df = pd.DataFrame(tokenized_split_dataset["test"])
        EA_df['predict'] = true_predictions
        EA_df['gold'] = true_labels
        EA_df.to_csv(RESULT_FOLDER+"error_analysis_"+task+"_"+model_name+'.csv')

    if not keep_models:
        shutil.rmtree(MODELS_FOLDER+model_name+"-finetuned-"+task)
    
    return metric.compute(predictions=true_predictions, references=true_labels)


def run_crossvalid(task,base_dataset,checkpoint,label_list,weighted=False,verbose=True,error_analysis=False,keep_models=False):
    
    #tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint, max_len=512,add_prefix_space=True)
    
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    results = {'fs':[],'prec':[],'rec':[]}
    for i in range(1,11):
        print(i)
        split_ds = datasets.DatasetDict({
            'train': base_dataset.filter(lambda example: example["fold"] not in [i,(i%10+1)]),
            'test': base_dataset.filter(lambda example: example["fold"] == i),
            'valid': base_dataset.filter(lambda example: example["fold"] == i%10+1)
        })
        print(label_list)
        res = run_one_fold(task,i,split_ds,checkpoint,tokenizer,label_list,weighted,verbose,error_analysis,keep_models)
        print(res)
        results['fs'].append(res['overall_f1'])
        results['prec'].append(res['overall_precision'])
        results['rec'].append(res['overall_recall'])
    return results


def run_complete_expe(expe_name,ds,label_list,weighted=False):
    #ds = ds.remove_columns(col_to_remove)
    print('====')
    print(expe_name)
    print('====')    
       
    m_name = re.sub("(\./|/)", "_", checkpoint)
    print('running ' + str(m_name))
    res_cv = run_crossvalid(expe_name,ds,checkpoint,label_list,
                                weighted=weighted,verbose=False,error_analysis=True,keep_models=False)
    
    
    res_cv_df = pd.DataFrame(res_cv)
    res_cv_df['model'] = m_name
    res_cv_df.to_csv(RESULT_FOLDER+expe_name+'_'+m_name+'_cv.csv')

    
    return 0

if args.task == 'prom':
    run_complete_expe('prom'+overall_name,dataset_prom,label_list_prom)
elif args.task == 'red':
    run_complete_expe('red'+overall_name,dataset_reduc,label_list_reduc)






