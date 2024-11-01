#! /bin/bash

#Batch Job Paremeters

#Operations

python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt models/wiki -task red

python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt models/spoken -task red

python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt models/babylm -task red

python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt FacebookAI/roberta-large -task red

python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt FacebookAI/roberta-base -task red

python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt models/wiki -task prom

python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt models/spoken -task prom

python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt models/babylm -task prom

python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt FacebookAI/roberta-large -task prom

python FineTune_and_Evaluate_on_Benchmarks_en.py -ckpt FacebookAI/roberta-base -task prom