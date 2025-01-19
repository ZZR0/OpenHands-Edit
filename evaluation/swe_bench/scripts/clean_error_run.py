import argparse
import json
import os

import re
import shutil

log_content = """
2025-01-11 00:05:11,038 - INFO - **RunRegressionObservation (source=None, exit code=0)**
Content:
Some tests failed.
"""
def get_regression_result(log_content):
    # 使用正则表达式匹配所有结果
    matches = re.findall(r'\*\*RunRegressionObservation.*?\*\*\nContent:\n(.*)', log_content)
    if matches is not None and len(matches) > 1:
        return matches, matches[0], matches[-1]
    return [], "", ""

def clean_by_error(args):
    run_dir = args.run_dir
    output_file = os.path.join(run_dir, 'output.jsonl')
    output_log_dir = os.path.join(run_dir, 'infer_logs')
    output_completion_dir = os.path.join(run_dir, 'llm_completions')

    output_instances = []
    with open(output_file, 'r') as f:
        for line in f:
            try:
                instance = json.loads(line)
                output_instances.append(instance)
            except Exception as e:
                print(f'Error loading instance: {e}')
    
    remove_instance_ids, remove_instance_ids_reason = [], []
    for instance in output_instances:
        instance_id = instance['instance_id']
        log_file = f'instance_{instance_id}.log'
        if not os.path.exists(os.path.join(output_log_dir, log_file)):
            print(f'{log_file} not found')
            continue
        
        with open(os.path.join(output_log_dir, log_file), 'r') as f:
            log_content = f.read()
            regression_result, first_regression_result, last_regression_result = get_regression_result(log_content)
            remove = None
            count_repeat_error = log_content.count('for the repeate')
            
            if 'Max retries exceeded with url:' in log_content:
                remove = 'Max retries exceeded with url:'
            elif 'VertexAIException BadRequestError' in log_content:
                remove = 'VertexAIException BadRequestError'
            # elif 'AnthropicException' in log_content:
            #     remove = 'AnthropicException BadRequestError'
            elif 'Attempt #12' in log_content:
                remove = 'Attempt #12'
            elif 'Error during action execution:' in log_content:
                remove = 'Error during action execution:'
            elif 'Agent got stuck in a loop' in log_content:
                remove = 'Agent got stuck in a loop'
            elif 'GLOBAL STEP 49' in log_content:
                remove = 'GLOBAL STEP 49'
            elif count_repeat_error > 0:
                remove = 'Repeate Error'
            elif 'Some tests failed.' in last_regression_result and 'All tests passed.' in first_regression_result:
                remove = 'Some tests failed.'
            elif 'error' in instance and instance['error'] is not None:
                remove = instance['error']
            # elif 'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.' in log_content:
            #     remove = 'ASK FOR HUMAN HELP.'
            
            if remove is not None:
                instance_id = log_file.split('.')[0].replace('instance_', '')
                print(f'{instance_id} is removed because {remove}')
                remove_instance_ids.append(instance_id)
                remove_instance_ids_reason.append(remove)
                
    report_file = os.path.join(run_dir, 'report.json')
    if os.path.exists(report_file):
        print("="*100)
        with open(report_file, 'r') as f:
            report = json.load(f)
        for instance_id, reason in zip(remove_instance_ids, remove_instance_ids_reason):
            if instance_id in report['resolved_ids']:
                print(f'{instance_id} is resolved but it is removed because {reason}')
        print("="*100)
    
    error_instance_ids = []
    empty_patch_instance_ids = []
    new_output_instances = []
    for instance in output_instances:
        if instance['instance_id'] in remove_instance_ids:
            error_instance_ids.append(instance['instance_id'])
            continue
        if instance['test_result']['git_patch'] == '':
            empty_patch_instance_ids.append(instance['instance_id'])
            continue
        new_output_instances.append(instance)

    with open(output_file, 'w') as f:
        for instance in new_output_instances:
            f.write(json.dumps(instance) + '\n')
            
    for log_file in os.listdir(output_log_dir):
        instance_id = log_file.split('.')[0].replace('instance_', '')
        if instance_id not in [instance['instance_id'] for instance in new_output_instances]:
            # print(f'{log_file} is removed')
            os.remove(os.path.join(output_log_dir, log_file))
                
    for log_dir in os.listdir(output_completion_dir):
        if log_dir not in [instance['instance_id'] for instance in new_output_instances]:
            # print(f'{log_dir} is removed')
            shutil.rmtree(os.path.join(output_completion_dir, log_dir))

    print(f'Removed {len(remove_instance_ids)} error instances: {remove_instance_ids}')
    print(
        f'Removed {len(empty_patch_instance_ids)} empty patch instances: {empty_patch_instance_ids}'
    )

def clean_by_result(args):
    run_dir = args.run_dir
    output_file = os.path.join(run_dir, 'output.jsonl')
    output_log_dir = os.path.join(run_dir, 'infer_logs')
    output_completion_dir = os.path.join(run_dir, 'llm_completions')

    output_instances = []
    with open(output_file, 'r') as f:
        for line in f:
            try:
                instance = json.loads(line)
                output_instances.append(instance)
            except Exception as e:
                print(f'Error loading instance: {e}')
    
    report_file = os.path.join(run_dir, 'report.json')
    if os.path.exists(report_file):
        with open(report_file, 'r') as f:
            report = json.load(f)
        resolved_ids = report['resolved_ids']
    else:
        print("Report file not found")
        return
    
    remove_instance_ids, remove_instance_ids_reason = [], []
    for instance in output_instances:
        instance_id = instance['instance_id']
        if instance_id not in resolved_ids:
            remove_instance_ids.append(instance_id)
            remove_instance_ids_reason.append('Not resolved')
    
    error_instance_ids = []
    new_output_instances = []
    for instance in output_instances:
        if instance['instance_id'] in remove_instance_ids:
            error_instance_ids.append(instance['instance_id'])
            continue
        new_output_instances.append(instance)

    with open(output_file, 'w') as f:
        for instance in new_output_instances:
            f.write(json.dumps(instance) + '\n')
            
    for log_file in os.listdir(output_log_dir):
        instance_id = log_file.split('.')[0].replace('instance_', '')
        if instance_id not in [instance['instance_id'] for instance in new_output_instances]:
            # print(f'{log_file} is removed')
            os.remove(os.path.join(output_log_dir, log_file))
                
    for log_dir in os.listdir(output_completion_dir):
        if log_dir not in [instance['instance_id'] for instance in new_output_instances]:
            # print(f'{log_dir} is removed')
            shutil.rmtree(os.path.join(output_completion_dir, log_dir))

    print(f'Removed {len(remove_instance_ids)} error instances: {remove_instance_ids}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, required=True)
    args = parser.parse_args()

    print(args)

    clean_by_error(args)
    # clean_by_result(args)
