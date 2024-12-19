import argparse
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, required=True)
    args = parser.parse_args()

    print(args)

    run_dir = args.run_dir
    output_file = os.path.join(run_dir, 'output.jsonl')
    output_log_dir = os.path.join(run_dir, 'infer_logs')

    remove_instance_ids = []
    for log_file in os.listdir(output_log_dir):
        if log_file.endswith('.log'):
            with open(os.path.join(output_log_dir, log_file), 'r') as f:
                log_content = f.read()
                if 'Max retries exceeded with url:' in log_content:
                    os.remove(os.path.join(output_log_dir, log_file))
                    remove_instance_ids.append(
                        log_file.split('.')[0].replace('instance_', '')
                    )
                elif 'VertexAIException BadRequestError' in log_content:
                    os.remove(os.path.join(output_log_dir, log_file))
                    remove_instance_ids.append(
                        log_file.split('.')[0].replace('instance_', '')
                    )
                elif 'Attempt #4' in log_content:
                    os.remove(os.path.join(output_log_dir, log_file))
                    remove_instance_ids.append(
                        log_file.split('.')[0].replace('instance_', '')
                    )

    error_instance_ids = []
    empty_patch_instance_ids = []
    output_instances = []
    with open(output_file, 'r') as f:
        for line in f:
            instance = json.loads(line)
            if instance['instance_id'] in remove_instance_ids:
                error_instance_ids.append(instance['instance_id'])
                continue
            if instance['test_result']['git_patch'] == '':
                empty_patch_instance_ids.append(instance['instance_id'])
                continue
            output_instances.append(instance)

    with open(output_file, 'w') as f:
        for instance in output_instances:
            f.write(json.dumps(instance) + '\n')

    print(f'Removed {len(remove_instance_ids)} error instances: {remove_instance_ids}')
    print(
        f'Removed {len(empty_patch_instance_ids)} empty patch instances: {empty_patch_instance_ids}'
    )
