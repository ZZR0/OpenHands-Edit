#!/bin/bash

tests_file=$1
dataset_name=$2

cd /hdd1/zzr/libro-agentless

python agentless/libro/reproduce_eval.py \
    --dataset $dataset_name \
    --split test\
    --tests_file $tests_file \
    --num_threads 10 \
    --output_folder results/test_agent \
    --skip_existing \
    --run_id test_agent

output_folder=$(dirname $tests_file)
rm -rf $output_folder/test_results
mv results/test_agent $output_folder/test_results
