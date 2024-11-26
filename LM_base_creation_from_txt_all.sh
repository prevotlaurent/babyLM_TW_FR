#! /bin/bash

#Batch Job Paremeters

#Operations

#python LM_base_creation_from_txt_spxlm_base.py -language en -corpus_name wiki -epoch 1

#python LM_base_creation_from_txt_spxlm_base.py -language en -corpus_name spoken -epoch 1

#python LM_base_creation_from_txt_spxlm_base.py -language fr -corpus_name wiki -epoch 1

#python LM_base_creation_from_txt_spxlm_base.py -language fr -corpus_name spoken -epoch 1

#python LM_base_creation_from_txt_spxlm_base.py -language zh -corpus_name wiki -epoch 1

python LM_base_creation_from_txt_spxlm_base.py -language cn -corpus_name spoken -epoch 1


#python LM_base_creation_from_txt_en.py -model_name babylm -epoch 1	

# python eval.py -check_point roberta-base -save_name roberta-base -batch_size 16

# python eval.py -check_point ./models/babylm_spoken_repp/ -save_name babylm_spoken_repp -batch_size 16

# python eval.py -check_point ./models/babylm_written_repp/ -save_name babylm_written_repp -batch_size 16

# python eval.py -check_point ./models/babylm_allbaby_repp/ -save_name babylm_allbaby_repp -batch_size 16



