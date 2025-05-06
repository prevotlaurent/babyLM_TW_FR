#! /bin/bash

#Batch Job Paremeters

#Operations

cd ../../../work/sftwang/babyLM_TW_FR/

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/en_babylm_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/en_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge en -corpus buckeye -benchmark_file 'data_benchmark/benchmark_en_updated_small.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/en_spoken_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/zh_mixed_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/zh_wiki_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_reduc_zh.csv' -results_dir './results_0126/' -red_cutoff 0.6 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge zh -corpus mcdc -benchmark_file 'data_benchmark/benchmark_prom_zh.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/zh_spoken_sp_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr1e-4_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 1e-4 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr1e-4_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr8e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 8e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr8e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr6e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 6e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr6e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr4e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 4e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr4e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/pruned_xlm-roberta-large'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/fr_mixed_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/fr_wiki_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 32 -ft_eps 10 -exp_tag standard_bs32_lr2e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task red -label_column red -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_reduc_fr_simple.csv' -results_dir './results_0126/' -red_cutoff 0.5 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'

python FineTune_and_Evaluate_on_Benchmarks_all.py -task prom -label_column prom -lge fr -corpus cid -benchmark_file 'data_benchmark/benchmark_prom_fr_simple.csv' -results_dir './results_0126/' -prom_cutoff 1.25 -freeze_to 0.0 -lr 2e-5 -batch_size 16 -ft_eps 10 -exp_tag standard_bs16_lr2e-5_ep10_f0.0_1 -ckpt './models/fr_conv_spbpe_concat'