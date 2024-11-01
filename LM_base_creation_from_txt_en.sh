#! /bin/bash

#Batch Job Paremeters

#Operations

python LM_base_creation_from_txt_en.py -model_name wiki

python LM_base_creation_from_txt_en.py -model_name spoken

python LM_base_creation_from_txt_en.py -model_name babylm

# python eval.py -check_point roberta-base -save_name roberta-base -batch_size 16

# python eval.py -check_point ./models/babylm_spoken_repp/ -save_name babylm_spoken_repp -batch_size 16

# python eval.py -check_point ./models/babylm_written_repp/ -save_name babylm_written_repp -batch_size 16

# python eval.py -check_point ./models/babylm_allbaby_repp/ -save_name babylm_allbaby_repp -batch_size 16



