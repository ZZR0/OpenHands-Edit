import json
import os


def analysis_reproduce():
    result_dict = {
        'agentless15_gpt4o': '/hdd2/zzr/experiments/evaluation/lite/20241028_agentless-1.5_gpt4o/results/results.json',
        'agentless15_claude': '/hdd2/zzr/experiments/evaluation/lite/20241202_agentless-1.5_claude-3.5-sonnet-20241022/results/results.json',
        'codeshelltester': '/hdd2/zzr/experiments/evaluation/lite/20241111_codeshelltester_gpt4o/results/results.json',
        'composio': '/hdd2/zzr/experiments/evaluation/lite/20241030_composio_swekit/results/results.json',
        'codeact': '/hdd2/zzr/experiments/evaluation/lite/20241025_OpenHands-CodeAct-2.1-sonnet-20241022/results/results.json',
        'ours-1': '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_best4_s5_iter20-run_1/report.json',
        'ours-2': '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_best4_s5_iter20-run_2/report.json',
        'ours-3': '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_best4_s5_iter20-run_3/report.json',
    }
    
    result = json.load(open(result_dict['codeact'], 'r'))
    strong_resolved_ids = result['resolved'] if 'resolved' in result else result['resolved_ids']
    result = json.load(open(result_dict['ours-2'], 'r'))
    our_resolved_ids = result['resolved'] if 'resolved' in result else result['resolved_ids']
    # analysis_dir = "/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_iter20-run_1/infer_logs"
    # analysis_dir = '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_iter20_3-run_1/infer_logs'
    analysis_dir = '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgentEdit/claude-3-5-sonnet-20241022_maxiter_100_N_v2.1-no-hint-l14-run_1/infer_logs'

    files = os.listdir(analysis_dir)
    files.sort()
    file_list = []
    easy_file_list = []
    for file in files:
        if file.endswith('.log'):
            with open(os.path.join(analysis_dir, file), 'r') as f:
                content = f.read()
                if 'COMMAND:\npython /workspace/reproduce_error.py' in content:
                    print(file)
                    file_list.append(file)
                    instance_ids = file.replace('.log', '').replace('instance_', '')
                    if instance_ids in strong_resolved_ids and instance_ids not in our_resolved_ids:
                        easy_file_list.append(file)
    print(len(file_list))
    print(easy_file_list)
    print(len(easy_file_list))
    
def analysis_verified_fix():
    # selected_ids = '/hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config-v-500.toml'
    # selected_ids = '/hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config-v-52.toml'
    selected_ids = '/hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config-v-rd-100.toml'
    # selected_ids = '/hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config-v-ez-100.toml'
    selected_ids = eval(open(selected_ids, 'r').read().replace('selected_ids = ', ''))

    result_dict = {
        'autocoderover': '/hdd2/zzr/experiments/evaluation/verified/20240628_autocoderover-v20240620/results/results.json',
        'marscode': '/hdd2/zzr/experiments/evaluation/verified/20241125_marscode-agent-dev/results/results.json',
        'solver': '/hdd2/zzr/experiments/evaluation/verified/20241028_solver/results/results.json',
        'gru': '/hdd2/zzr/experiments/evaluation/verified/20240824_gru/results/results.json',
    }
    
    for key, value in result_dict.items():
        result = json.load(open(value, 'r'))
        resolved_ids = []
        for id in selected_ids:
            if id in result['resolved']:
                resolved_ids.append(id)
        print(key, len(resolved_ids), len(selected_ids), len(resolved_ids) / len(selected_ids))
    
def analysis_lite_fix():
    selected_ids = [  # type: ignore
        "django__django-12284",
        "django__django-11910",
        "django__django-12125",
        "django__django-11999",
        "django__django-11848",
        "django__django-11630",
        "django__django-11283",
        "django__django-11049",
        "django__django-10914",
        "astropy__astropy-14365",
        "astropy__astropy-14995",
        "astropy__astropy-7746",
        "django__django-11019",
        "django__django-11001",
        "django__django-11179",
        "django__django-11422",
        "astropy__astropy-6938",
        "astropy__astropy-14182",
        "django__django-11564",
        "django__django-11039",
        "astropy__astropy-12907",
        "django__django-10924",
        "django__django-11583",
        "django__django-11620",
        "django__django-11905",
        "django__django-11797",
        "django__django-11964",
        "django__django-11742",
        "django__django-11815",
        "django__django-12453",
        "django__django-12589",
        "django__django-12700",
        "django__django-13230",
        "django__django-12113",
        "django__django-12708",
        "django__django-12856",
        "django__django-13033",
        "django__django-13315",
        "django__django-13551",
        "django__django-13321",
        "django__django-13660",
        "django__django-13933",
        "django__django-13925",
        "django__django-13710",
        "django__django-14017",
        "django__django-13964",
        "django__django-14155",
        "django__django-14238",
        "django__django-14730",
        "django__django-14534",
        "django__django-14787",
        "django__django-14752",
        "django__django-14855",
        "django__django-14915",
        "django__django-15252",
        "django__django-14999",
        "django__django-15202",
        "django__django-15320",
        "django__django-15498",
        "django__django-15781",
        "django__django-15789",
        "django__django-15814",
        "django__django-15388",
        "django__django-15738",
        "django__django-15996",
        "django__django-16041",
        "django__django-15819",
        "django__django-16229",
        "django__django-16046",
        "django__django-16527",
        "django__django-16408",
        "django__django-16595",
        "django__django-16820",
        "django__django-17051",
        "django__django-17087",
        "matplotlib__matplotlib-18869",
        "matplotlib__matplotlib-23476",
        "matplotlib__matplotlib-22711",
        "matplotlib__matplotlib-23299",
        "matplotlib__matplotlib-23563",
        "matplotlib__matplotlib-23964",
        "matplotlib__matplotlib-23987",
        "matplotlib__matplotlib-24334",
        "matplotlib__matplotlib-25311",
        "matplotlib__matplotlib-25442",
        "mwaskom__seaborn-3010",
        "mwaskom__seaborn-3190",
        "matplotlib__matplotlib-25498",
        "matplotlib__matplotlib-26011",
        "mwaskom__seaborn-2848",
        "psf__requests-2674",
        "pallets__flask-4045",
        "pallets__flask-4992",
        "pallets__flask-5063",
        "mwaskom__seaborn-3407",
        "matplotlib__matplotlib-25332",
        "psf__requests-1963",
        "psf__requests-2148",
        "matplotlib__matplotlib-23314",
        "matplotlib__matplotlib-24149",
    ]
    
    result_dict = {
        'agentless15_gpt4o': '/hdd2/zzr/experiments/evaluation/lite/20241028_agentless-1.5_gpt4o/results/results.json',
        'agentless15_claude': '/hdd2/zzr/experiments/evaluation/lite/20241202_agentless-1.5_claude-3.5-sonnet-20241022/results/results.json',
        'codeshelltester': '/hdd2/zzr/experiments/evaluation/lite/20241111_codeshelltester_gpt4o/results/results.json',
        'composio': '/hdd2/zzr/experiments/evaluation/lite/20241030_composio_swekit/results/results.json',
        'codeact': '/hdd2/zzr/experiments/evaluation/lite/20241025_OpenHands-CodeAct-2.1-sonnet-20241022/results/results.json',
        'ours-1': '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_best4_s5_iter20-run_1/report.json',
        'ours-2': '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_best4_s5_iter20-run_2/report.json',
        'ours-3': '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-reproduce_best4_s5_iter20-run_3/report.json',
    }
    
    resolved_dict = {
        'agentless15_gpt4o': [],
        'agentless15_claude': [],
        'codeshelltester': [],
        'composio': [],
        'codeact': [],
        'ours-1': [],
        'ours-2': [],
        'ours-3': [],
    }
    
    for key, value in result_dict.items():
        result = json.load(open(value, 'r'))
        resolved_ids = result['resolved'] if 'resolved' in result else result['resolved_ids']
        sub_resolved_ids = []
        for id in selected_ids:
            if id in resolved_ids:
                sub_resolved_ids.append(id)
        resolved_dict[key] = sub_resolved_ids
        print(key, len(sub_resolved_ids), len(selected_ids), len(sub_resolved_ids) / len(selected_ids))
    
    resolved_dict['ours'] = list(set(resolved_dict['ours-1']) | set(resolved_dict['ours-2']) | set(resolved_dict['ours-3']))
    resolved_dict['combine'] = list(set(resolved_dict['agentless15_gpt4o']) | set(resolved_dict['agentless15_claude']) | set(resolved_dict['composio']) | set(resolved_dict['codeact']) | set(resolved_dict['codeshelltester']))
    print(set(resolved_dict['codeact']) - set(resolved_dict['ours']))
    print(set(resolved_dict['ours']) - set(resolved_dict['codeact']))
    print(set(resolved_dict['combine']) - set(resolved_dict['ours']))
    print(set(resolved_dict['ours']) - set(resolved_dict['combine']))
    
def analysis_merge_result():
    result_list = [
        '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_50_N_v2.1-no-hint-selectiontest_strreplace_view_v1-run_1/report.json',
        '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-selectiontest_strreplace_view_v1-run_1/report.json',
        '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-selectiontest_strreplace_view_v1-run_1/report.json',
    ]
    resolved_ids = set()
    for result in result_list:
        result = json.load(open(result, 'r'))
        resolved_ids.update(result['resolved_ids'])
    print(len(resolved_ids))

def analysis_step():
    from statistics import mean
    def get_step_count(log_dir, sids):
        step_count_list = []
        for id in sids:
            log_file = f'instance_{id}.log'
            with open(os.path.join(log_dir, log_file), 'r') as f:
                content = f.read()
                step_count = content.count('GLOBAL STEP')
                step_count_list.append(step_count)
        return step_count_list
        
    result_dir = '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/claude-3-5-sonnet-20241022_maxiter_100_N_v2.1-no-hint-selectiontest-run_1/'
    report_path = os.path.join(result_dir, 'report.json')
    report = json.load(open(report_path, 'r'))
    
    resolved_ids = report['resolved_ids']
    unresolved_ids = report['unresolved_ids']
    log_dir = os.path.join(result_dir, 'infer_logs')
    
    step_count_list = get_step_count(log_dir, resolved_ids)
    print("resolved_ids", len(step_count_list), mean(step_count_list), max(step_count_list), min(step_count_list))
    step_count_list = get_step_count(log_dir, unresolved_ids)
    print("unresolved_ids", len(step_count_list), mean(step_count_list), max(step_count_list), min(step_count_list))

def sample():
    import random
    # result_dict = {
    #     'autocoderover': '/hdd2/zzr/experiments/evaluation/verified/20240628_autocoderover-v20240620/results/results.json',
    #     'marscode': '/hdd2/zzr/experiments/evaluation/verified/20241125_marscode-agent-dev/results/results.json',
    #     'solver': '/hdd2/zzr/experiments/evaluation/verified/20241028_solver/results/results.json',
    #     'gru': '/hdd2/zzr/experiments/evaluation/verified/20240824_gru/results/results.json',
    # }
    # all_resolved_ids = []
    # for key, value in result_dict.items():
    #     result = json.load(open(value, 'r'))
    #     all_resolved_ids.extend(result['resolved'])
    
    all_resolved_ids = []
    
    verified_ids = '/hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config-v-500.toml'
    verified_ids = eval(open(verified_ids, 'r').read().replace('selected_ids = ', ''))
    all_resolved_ids = verified_ids
    
    all_resolved_ids = set(all_resolved_ids)
    print(len(all_resolved_ids))
    print(random.sample(list(all_resolved_ids), 100))

def check_image_exist():
    verified_ids = '/hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/config-v-500.toml'
    verified_ids = eval(open(verified_ids, 'r').read().replace('selected_ids = ', ''))
    images_dict = json.load(open('/hdd2/zzr/OpenHands-fn-calling/evaluation/swe_bench/scripts/docker/images_dict.json', 'r'))
    for key, value in images_dict.items():
        if key in verified_ids:
            if not os.path.exists(value['image_path']):
                print(key, value['image_path'])

def analysis_output():
    result_dir = '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-selectiontest_strreplace_view_v1-run_1/'
    output_path = os.path.join(result_dir, 'output.jsonl')
    output = open(output_path, 'r').readlines()
    for line in output:
        line = json.loads(line)
        if 'error' in line and line['error'] is not None:
            print(line['instance_id'], line['error'])
        # exit()


if __name__ == '__main__':
    # analysis_reproduce()
    # analysis_verified_fix()
    # analysis_lite_fix()
    # analysis_merge_result()
    # analysis_step()
    # sample()
    # check_image_exist()
    analysis_output()