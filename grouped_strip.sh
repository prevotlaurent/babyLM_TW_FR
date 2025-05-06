#! /bin/bash

#Batch Job Paremeters

#Operations

cd ../../../work/sftwang/babyLM_TW_FR/

python LM_base_creation_from_txt_spxlm_base_V2.py -language fr -corpus_name conv -epoch 100 -group_texts True -exp_tag 1.5e4 -patience 10 -learning_rate 1.5e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language fr -corpus_name wiki -epoch 100 -group_texts True -exp_tag 1.5e4 -patience 10 -learning_rate 1.5e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language fr -corpus_name mixed -epoch 100 -group_texts True -exp_tag 1.5e4 -patience 10 -learning_rate 1.5e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language fr -corpus_name conv -epoch 100 -group_texts True -exp_tag 2e4 -patience 10 -learning_rate 2e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language fr -corpus_name wiki -epoch 100 -group_texts True -exp_tag 2e4 -patience 10 -learning_rate 2e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language fr -corpus_name mixed -epoch 100 -group_texts True -exp_tag 2e4 -patience 10 -learning_rate 2e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language fr -corpus_name conv -epoch 100 -group_texts True -exp_tag 1e4 -patience 10 -learning_rate 1e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language fr -corpus_name wiki -epoch 100 -group_texts True -exp_tag 1e4 -patience 10 -learning_rate 1e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language fr -corpus_name mixed -epoch 100 -group_texts True -exp_tag 1e4 -patience 10 -learning_rate 1e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language en -corpus_name spoken -epoch 100 -group_texts True -exp_tag 1.5e4 -patience 10 -learning_rate 1.5e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language en -corpus_name wiki -epoch 100 -group_texts True -exp_tag 1.5e4 -patience 10 -learning_rate 1.5e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language en -corpus_name babylm -epoch 100 -group_texts True -exp_tag 1.5e4 -patience 10 -learning_rate 1.5e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language en -corpus_name spoken -epoch 100 -group_texts True -exp_tag 2e4 -patience 10 -learning_rate 2e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language en -corpus_name wiki -epoch 100 -group_texts True -exp_tag 2e4 -patience 10 -learning_rate 2e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language en -corpus_name babylm -epoch 100 -group_texts True -exp_tag 2e4 -patience 10 -learning_rate 2e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language en -corpus_name spoken -epoch 100 -group_texts True -exp_tag 1e4 -patience 10 -learning_rate 1e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language en -corpus_name wiki -epoch 100 -group_texts True -exp_tag 1e4 -patience 10 -learning_rate 1e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language en -corpus_name babylm -epoch 100 -group_texts True -exp_tag 1e4 -patience 10 -learning_rate 1e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name spoken -epoch 200 -group_texts True -exp_tag 1.5e4 -patience 10 -learning_rate 1.5e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name wiki -epoch 200 -group_texts True -exp_tag 1.5e4 -patience 10 -learning_rate 1.5e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name mixed -epoch 200 -group_texts True -exp_tag 1.5e4 -patience 10 -learning_rate 1.5e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name spoken -epoch 200 -group_texts True -exp_tag 2e4 -patience 10 -learning_rate 2e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name wiki -epoch 200 -group_texts True -exp_tag 2e4 -patience 10 -learning_rate 2e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name mixed -epoch 200 -group_texts True -exp_tag 2e4 -patience 10 -learning_rate 2e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name spoken -epoch 200 -group_texts True -exp_tag 1e4 -patience 10 -learning_rate 1e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name wiki -epoch 200 -group_texts True -exp_tag 1e4 -patience 10 -learning_rate 1e-4

python LM_base_creation_from_txt_spxlm_base_V2.py -language zh -corpus_name mixed -epoch 200 -group_texts True -exp_tag 1e4 -patience 10 -learning_rate 1e-4