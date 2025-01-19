#!/bin/bash

function run_one() {
    echo "CLEAN ERROR RUN"
    python /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/scripts/clean_error_run.py --run-dir /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_50_N_v2.1-no-hint-selectiontest_strreplace_regression_mem_temp0.6-run_1/

    echo "RUN INFER"
    EXP_NAME="selectiontest_strreplace_regression_mem_temp0.6" ./evaluation/swe_bench/scripts/run_infer.sh llm.gemini-2.0-flash-exp HEAD CodeActAgentEdit 500 50 12 'princeton-nlp/SWE-bench_Verified' 'test' 1

    echo "EVAL INFER"
    # 定义要运行的命令
    command="./evaluation/swe_bench/scripts/eval_infer.sh /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_50_N_v2.1-no-hint-selectiontest_strreplace_regression_mem_temp0.6-run_1/output.jsonl princeton-nlp/SWE-bench_Verified test"
    # output=$($command 2>&1)
    # echo "$output"

    # 循环执行命令，直到输出包含 "resolved_ids"
    while true; do
        output=$(yes y | $command 2>&1)

        # 打印输出（可选）
        echo "$output"

        # 检查输出中是否包含 "resolved_ids"
        if echo "$output" | grep -q "Instances resolved:"; then
            echo "找到 'resolved_ids'，脚本停止运行。"
            break
        fi

        echo "未找到 'resolved_ids'，重新运行命令..."
    done

    
}

# echo "COPY CONFIG"
# rm /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config.toml
# cp /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config-v-0-125.toml /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config.toml

# # run_one
# run_one
# run_one
# run_one
# run_one

# echo "MOVE OUTPUT"
# mv /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit-v-0-125


# echo "CLEAN DOCKER"
# docker container ls | grep openhands | awk '{print $1}' | xargs docker rm -f
# yes y | docker builder prune

# echo "DONE config-v-0-125"



# echo "COPY CONFIG"
# rm /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config.toml
# cp /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config-v-1-125.toml /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config.toml

# run_one
# run_one
# run_one
# run_one
# run_one

# echo "MOVE OUTPUT"
# mv /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit-v-1-125


# echo "CLEAN DOCKER"
# docker container ls | grep openhands | awk '{print $1}' | xargs docker rm -f
# yes y | docker builder prune

# echo "DONE config-v-1-125"



# echo "COPY CONFIG"
# rm /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config.toml
# cp /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config-v-2-125.toml /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config.toml

# run_one
# run_one
# run_one
# run_one
# run_one

# echo "MOVE OUTPUT"
# mv /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit-v-2-125


# echo "CLEAN DOCKER"
# docker container ls | grep openhands | awk '{print $1}' | xargs docker rm -f
# yes y | docker builder prune

# echo "DONE config-v-2-125"





echo "COPY CONFIG"
rm /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config.toml
cp /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config-v-3-125.toml /hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config.toml

# run_one
run_one
run_one
run_one
run_one

echo "MOVE OUTPUT"
mv /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit-v-3-125


echo "CLEAN DOCKER"
docker container ls | grep openhands | awk '{print $1}' | xargs docker rm -f
yes y | docker builder prune

echo "DONE config-v-3-125"