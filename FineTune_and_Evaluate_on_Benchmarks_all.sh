#! /bin/bash

#Batch Job Paremeters

#Operations






#python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt models/wiki -task red

python FineTune_and_Evaluate_on_Benchmarks_all.py -task bc -corpus cid -benchmark_file 'data_benchmark/benchmark_bc_fr.csv' -ckpt 'models/fr_spoken_spbpe' -label_column bc

#python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr.csv' -ckpt 'models/fr_spoken_spbpe' -label_column red

#python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr.csv' -ckpt 'models/fr_spoken_spbpe' -label_column prom -task prom

#python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt models/babylm -task red

#python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt FacebookAI/roberta-large -task red

#python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt FacebookAI/roberta-base -task red

#python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt models/wiki -task prom

#python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt models/spoken -task prom

#python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt models/babylm -task prom

#python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt FacebookAI/roberta-large -task prom

#python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt FacebookAI/roberta-base -task prom
