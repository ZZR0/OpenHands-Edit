import json
import os
from multiprocessing import Pool, cpu_count

from datasets import load_dataset
from tqdm import tqdm

from openhands.core.config import (
    get_llm_config_arg,
    get_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.llm.llm import LLM

SYSTEM_PROMPT = (
    'You are a pull request reviewer. You need to choose the one PR from multiple that'
    ' actually will resolve the given issue.'
)


def select_patch(
    llm: LLM, instance: dict, patches: list[dict]
) -> tuple[int, dict, list]:
    # import pdb; pdb.set_trace()
    # return 0, patches[0]['test_result']['git_patch'], ''
    messages = []
    messages.append(Message(role='system', content=[TextContent(text=SYSTEM_PROMPT)]))

    init_prompt = f"Here is the issue: <issue>{instance['problem_statement']}</issue>.\n\nFirst, please analyze the root cause of the issue.\n"
    messages.append(Message(role='user', content=[TextContent(text=init_prompt)]))
    response = llm.completion(
        messages=llm.format_messages_for_llm(messages), temperature=1.0, n=1
    )
    messages.append(
        Message(
            role='assistant',
            content=[TextContent(text=response.choices[0].message.content)],
        )
    )

    messages.append(
        Message(
            role='user', content=[TextContent(text='Analyze how to resolve the issue.')]
        )
    )
    response = llm.completion(
        messages=llm.format_messages_for_llm(messages), temperature=1.0, n=1
    )
    messages.append(
        Message(
            role='assistant',
            content=[TextContent(text=response.choices[0].message.content)],
        )
    )

    select_prompt = 'Here are some patches:\n'
    for idx, patch in enumerate(patches, start=1):
        select_prompt += f"Patch {idx}:\n{patch['test_result']['git_patch']}\n"

    # import pdb; pdb.set_trace()

    question = (
        'Based on your analysis, '
        'think about which patch best resolves the issue. Tell me the number of'
        ' the patch as well as the reason you choose it. Provide your answer in'
        ' the following json format:\n'
        '\n'
        '```json\n'
        '{\n'
        '    "patch_number": ...,\n'
        '    "reason": "..."'
        '}\n'
        '```\n'
        'where `patch_number` is one of the patch numbers, and reason is a string'
        ' stating the reason to your choice.\n\n'
        'NOTE: the patch should only do what is necessary to address the issue. If multiple'
        ' patches look reasonable, choose the one that makes the least changes.'
    )
    select_prompt += question
    messages.append(Message(role='user', content=[TextContent(text=select_prompt)]))

    indices = {}

    response = llm.completion(
        messages=llm.format_messages_for_llm(messages), temperature=1.0, n=3
    )
    for choice in response.choices:
        try:
            data = json.loads(
                choice.message.content.split('```json')[1].split('```')[0]
            )
            index = int(data['patch_number']) - 1
        except Exception:
            index = 0

        if index not in indices:
            indices[index] = 0
        indices[index] += 1

    index = max(indices, key=indices.get)
    print(index, indices)
    if index >= len(patches):
        raise RuntimeError('out-of-bound patch selection by LLM')

    return index, patches[index], messages


def prepare_dataset(instances: list, patches: list) -> list:
    patches_dict = {}
    for patch in patches:
        for p in patch:
            if p['instance_id'] not in patches_dict:
                patches_dict[p['instance_id']] = []
            patches_dict[p['instance_id']].append(p)

    new_instances = []
    for instance in instances:
        if instance['instance_id'] in patches_dict:
            instance['patches'] = patches_dict[instance['instance_id']]
            new_instances.append(instance)
    return new_instances


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench',
        help='data set to evaluate on, either full-test or lite-test',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    parser.add_argument(
        '--patch-files',
        type=str,
        nargs='+',
        help='Path to input patch files',
        required=True,
    )
    # import pdb; pdb.set_trace()
    args, _ = parser.parse_known_args()
    output_dir = (
        '-'.join(os.path.dirname(args.patch_files[0]).split('-')[:-1]) + '-select'
    )

    # NOTE: It is preferable to load datasets from huggingface datasets and perform post-processing
    # so we don't need to manage file uploading to OpenHands's repo
    instances = load_dataset(args.dataset, split=args.split)
    logger.info(f'Loaded dataset {args.dataset} with split {args.split}')
    instances = instances.to_list()
    # instances = filter_dataset(instances.to_pandas(), 'instance_id')
    patches = []
    for patch_path in args.patch_files:
        with open(patch_path, 'r') as f:
            patches.append([json.loads(line) for line in f.readlines()])
    instances = prepare_dataset(instances, patches)

    llm_config = None
    # import pdb; pdb.set_trace()
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        llm_config.log_completions = True
        llm_config.log_completions_folder = os.path.join(output_dir, 'llm_completions')

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    llm = LLM(config=llm_config)

    details = {}

    dataset_descrption = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )

    output = []
    # for instance in tqdm(instances, desc='Selecting patches'):
    #     patch_idx, patch, _ = select_patch(llm, instance, instance['patches'])
    #     # patch['select_traj'] = ''
    #     output.append(patch)

    def select_patch_wrapper(item):
        instance, patches = item
        return select_patch(llm, instance, patches)

    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(
                    select_patch_wrapper,
                    [(instance, instance['patches']) for instance in instances],
                ),
                total=len(instances),
                desc='Selecting patches',
            )
        )

        for patch_idx, patch, thread in results:
            patch['select_traj'] = ''
            output.append(patch)

    with open(os.path.join(output_dir, 'output.jsonl'), 'w') as f:
        for item in output:
            f.write(json.dumps(item) + '\n')
