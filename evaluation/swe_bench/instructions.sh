export https_proxy=http://127.0.0.1:10809 # 注意只能设置 https_proxy，如果设置了 all 或者 http 会导致把 localhost 的请求也走代理
export no_proxy=localhost,127.0.0.1,local,.local
export EVAL_DOCKER_IMAGE_PREFIX=dockerpull.org/xingyaoww/

export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring # 如果 make build 之后就报错
poetry self update # 如果遇到了 Cannot install swebench 的报错



# ./evaluation/benchmarks/swe_bench/scripts/run_infer.sh [model_config] [git-version] [agent] [eval_limit] [max_iter] [num_workers] [dataset] [dataset_split] [N_RUNS]
# ./evaluation/benchmarks/swe_bench/scripts/run_infer.sh llm.claude-3-5-sonnet-20241022 HEAD CodeActAgent 1 30 1 princeton-nlp/SWE-bench_Verified test
# ./evaluation/benchmarks/swe_bench/scripts/run_infer.sh llm.claude-3-5-sonnet-20241022 HEAD CodeActAgent 500 100 10 'princeton-nlp/SWE-bench_Verified' 'test' 1
./evaluation/swe_bench/scripts/run_infer.sh llm.claude-3-5-sonnet-20241022 HEAD CodeActAgent 500 100 1 'princeton-nlp/SWE-bench_Verified' 'test' 1
./evaluation/swe_bench/scripts/run_infer.sh llm.gpt-4o-2024-05-13 HEAD CodeActAgent 500 100 1 'princeton-nlp/SWE-bench_Verified' 'test' 1
./evaluation/swe_bench/scripts/run_infer.sh llm.deepseek-chat HEAD CodeActAgent 500 100 1 'princeton-nlp/SWE-bench_Verified' 'test' 1

EXP_NAME="wtest" ./evaluation/swe_bench/scripts/run_infer.sh llm.gpt-4o-mini-2024-07-18-aigc HEAD CodeActAgentEdit 500 100 4 'princeton-nlp/SWE-bench_Verified' 'test' 1
./evaluation/swe_bench/scripts/run_infer.sh llm.gpt-4o-mini-2024-07-18-openai HEAD CodeActAgentEdit 500 100 1 'princeton-nlp/SWE-bench_Verified' 'test' 1

EXP_NAME="notest" ./evaluation/swe_bench/scripts/run_infer.sh llm.gemini-2.0-flash-exp HEAD CodeActAgentEdit 500 100 10 'princeton-nlp/SWE-bench_Verified' 'test' 1

EXP_NAME="notest_best8_iter20" ./evaluation/swe_bench/scripts/run_infer.sh llm.gemini-2.0-flash-exp HEAD CodeActAgentEdit 500 100 4 'princeton-nlp/SWE-bench_Verified' 'test' 1

./evaluation/swe_bench/scripts/run_infer.sh llm.gpt-4o-2024-05-13-aigc HEAD CodeActAgentEdit 500 100 4 'princeton-nlp/SWE-bench_Verified' 'test' 1
./evaluation/swe_bench/scripts/run_infer.sh llm.gpt-4o-2024-05-13-openai HEAD CodeActAgentEdit 500 100 4 'princeton-nlp/SWE-bench_Verified' 'test' 1
./evaluation/swe_bench/scripts/eval_infer.sh /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gpt-4o-2024-05-13_maxiter_100_N_v2.1-no-hint-run_1/output.jsonl "" "princeton-nlp/SWE-bench_Verified" "test"

./evaluation/swe_bench/scripts/run_infer.sh llm.gpt-4o-mini-2024-07-18 HEAD CodeActAgentEdit 500 100 4 'princeton-nlp/SWE-bench_Verified' 'test' 1
./evaluation/swe_bench/scripts/eval_infer.sh /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gpt-4o-mini-2024-07-18_maxiter_100_N_v2.1-no-hint-run_1/output.jsonl "" "princeton-nlp/SWE-bench_Verified" "test"

./evaluation/swe_bench/scripts/eval_infer.sh /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-4o-mini-2024-07-18_maxiter_100_N_v2.1-no-hint-run_1/output.jsonl "" "princeton-nlp/SWE-bench_Verified" "test"
./evaluation/swe_bench/scripts/eval_infer.sh /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gpt-4o-mini-2024-07-18_maxiter_100_N_v2.1-no-hint-run_1/output.jsonl "" "princeton-nlp/SWE-bench_Verified" "test"


./evaluation/swe_bench/scripts/run_infer.sh llm.gpt-4o-2024-05-13 HEAD CodeActAgent 500 100 1 'princeton-nlp/SWE-bench_Verified' 'test' 1

./evaluation/swe_bench/scripts/eval_infer.sh evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-4o-2024-05-13_maxiter_100_N_v2.2-no-hint-run_1/output.jsonl

./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh $YOUR_OUTPUT_JSONL [instance_id] [dataset_name] [split]

rm -r /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gpt-4o-mini-2024-07-18_maxiter_100_N_v2.1-no-hint-run_1

docker images | grep all-hands-ai | awk 'length($2) == 44' | awk '{print $3}' | xargs docker rmi
docker container ls | grep openhands-runtime- | awk '{print $1}' | xargs docker rm -f

docker images | grep all-hands-ai | awk 'length($2) == 27' | grep days | awk '{ print $1":"$2 }' | xargs -I {} docker save -o /hdd1/zzr/docker_images/{}.tar {}

poetry run python evaluation/swe_bench/run_select.py \
  --llm-config llm.gemini-2.0-flash-exp \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --patch-files /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_iter20_2-run_1/output.jsonl \
                /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_iter20_3-run_1/output.jsonl \
                /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_iter20_4-run_1/output.jsonl \
                /hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_iter20_5-run_1/output.jsonl

docker builder prune


./evaluation/swe_bench/scripts/eval_infer.sh $YOUR_OUTPUT_JSONL [instance_id] [dataset_name] [split]
./evaluation/swe_bench/scripts/eval_infer.sh $YOUR_OUTPUT_JSONL \ princeton-nlp/SWE-bench_Verified test


./evaluation/swe_bench/scripts/run_infer.sh llm.gpt-4o-2024-05-13 HEAD CodeActAgent 1 100 1 'princeton-nlp/SWE-bench_Verified' 'test' 1
