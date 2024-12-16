import os


def analysis_reproduce():
    # analysis_dir = "/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_iter20-run_1/infer_logs"
    analysis_dir = '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_iter20_3-run_1/infer_logs'

    files = os.listdir(analysis_dir)
    files.sort()
    for file in files:
        if file.endswith('.log'):
            with open(os.path.join(analysis_dir, file), 'r') as f:
                content = f.read()
                if 'COMMAND:\npython /workspace/reproduce_error.py' in content:
                    print(file)


if __name__ == '__main__':
    analysis_reproduce()
