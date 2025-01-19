
import json
import os

def load_jsonl(file_path):
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def save_jsonl(results, file_path):
    with open(file_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def merge_results():
    result_paths = [
        "/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit-0/gemini-2.0-flash-exp_maxiter_50_N_v2.1-no-hint-selectiontest_strreplace_regression_mem_temp0.6-run_1/output.jsonl",
        "/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit-1/gemini-2.0-flash-exp_maxiter_50_N_v2.1-no-hint-selectiontest_strreplace_regression_mem_temp0.6-run_1/output.jsonl",
        "/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit-2/gemini-2.0-flash-exp_maxiter_50_N_v2.1-no-hint-selectiontest_strreplace_regression_mem_temp0.6-run_1/output.jsonl",
        "/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit-3/gemini-2.0-flash-exp_maxiter_50_N_v2.1-no-hint-selectiontest_strreplace_regression_mem_temp0.6-run_1/output.jsonl"
    ]
    save_path = "/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeShellAgent/gemini-2.0-flash-exp/output.jsonl"
    merged_results = []
    all_preds = []
    for result_path in result_paths:
        results = load_jsonl(result_path)
        merged_results.extend(results)
        all_preds.extend([{"model_name_or_path": "codeshellagent", "instance_id": result["instance_id"], "model_patch": result["test_result"]["git_patch"]} for result in results])
    all_preds = sorted(all_preds, key=lambda x: x['instance_id'])
    merged_results = sorted(merged_results, key=lambda x: x['instance_id'])
    save_jsonl(merged_results, save_path)
    save_jsonl(all_preds, save_path.replace("output.jsonl", "all_preds.jsonl"))

def get_trajs():
    result_paths = [
        "/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit-0/gemini-2.0-flash-exp_maxiter_50_N_v2.1-no-hint-selectiontest_strreplace_regression_mem_temp0.6-run_1/llm_completions",
        "/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit-1/gemini-2.0-flash-exp_maxiter_50_N_v2.1-no-hint-selectiontest_strreplace_regression_mem_temp0.6-run_1/llm_completions",
        "/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit-2/gemini-2.0-flash-exp_maxiter_50_N_v2.1-no-hint-selectiontest_strreplace_regression_mem_temp0.6-run_1/llm_completions",
        "/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit-3/gemini-2.0-flash-exp_maxiter_50_N_v2.1-no-hint-selectiontest_strreplace_regression_mem_temp0.6-run_1/llm_completions"
    ]
    trajs = {}
    for result_path in result_paths:
        traj_dirs = os.listdir(result_path)
        for instance_traj_dir in traj_dirs:
            traj_files = os.listdir(os.path.join(result_path, instance_traj_dir))
            traj_files.sort(key=lambda x: float(x.split("-")[-1].split(".")[0]))
            if len(traj_files) == 0:
                trajs[instance_traj_dir] = []
                continue
            last_traj_file = traj_files[-1]
            last_traj_file_path = os.path.join(result_path, instance_traj_dir, last_traj_file)
            traj = load_json(last_traj_file_path)
            trajs[instance_traj_dir] = [{"role": msg["role"], "content": msg["content"][0]["text"]} for msg in traj["messages"]] + [{"role": "assistant", "content": traj["response"]["choices"][0]["message"]["content"]}]
    
    os.makedirs("/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeShellAgent/gemini-2.0-flash-exp/trajs", exist_ok=True)
    for instance_id, traj in trajs.items():
        save_json(traj, f"/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeShellAgent/gemini-2.0-flash-exp/trajs/{instance_id}.json")

if __name__ == '__main__':
    # merge_results()
    get_trajs()