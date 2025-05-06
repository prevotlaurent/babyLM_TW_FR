#! /bin/bash

#Batch Job Paremeters

#Operations


python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name spoken -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name wiki -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name mixed -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64

python LM_base_creation_from_txt_spxlm_base_V2.py -language fr -corpus_name conv -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64

python LM_base_creation_from_txt_spxlm_base_V2.py -language fr -corpus_name wiki -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64

python LM_base_creation_from_txt_spxlm_base_V2.py -language fr -corpus_name mixed -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name spoken -epoch 64 -group_texts True -exp_tag 1e3 -patience 10 -learning_rate 1e-3 -batch_size 64

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name wiki -epoch 64 -group_texts True -exp_tag 1e3 -patience 10 -learning_rate 1e-3 -batch_size 64

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name mixed -epoch 64 -group_texts True -exp_tag 1e3 -patience 10 -learning_rate 1e-3 -batch_size 64

