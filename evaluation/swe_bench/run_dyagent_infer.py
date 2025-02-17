import asyncio
import json
import os
import random
import re
import tempfile
import time
from datetime import datetime
from typing import Any, List

import docker
import pandas as pd
import toml
from datasets import load_dataset

import openhands.agenthub
from evaluation.swe_bench.prompt import CODEACT_SWE_PROMPT
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    codeact_user_response,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    AppConfig,
    SandboxConfig,
    get_llm_config_arg,
    get_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.core.message import Message, TextContent
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation
from openhands.events.serialization.event import event_to_dict
from openhands.llm.llm import LLM
from openhands.runtime.base import Runtime
from openhands.runtime.utils.runtime_build import (
    get_hash_for_lock_files,
    get_hash_for_source_files,
    get_runtime_image_repo_and_tag,
    oh_version,
)
from openhands.runtime.utils.shutdown_listener import sleep_if_should_continue
from openhands.utils.async_utils import call_async_from_sync

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
USE_INSTANCE_IMAGE = os.environ.get('USE_INSTANCE_IMAGE', 'false').lower() == 'true'
DOCKER_IMAGE_DIR = os.environ.get('DOCKER_IMAGE_DIR', '/hdd1/zzr/docker_images/')


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
    'CodeActSWEAgent': codeact_user_response,
    'CodeActAgentEdit': codeact_user_response,
}


def _get_swebench_workspace_dir_name(instance: pd.Series) -> str:
    return f'{instance.repo}__{instance.version}'.replace('/', '__')


def get_instruction(instance: pd.Series, metadata: EvalMetadata):
    llm = LLM(config=metadata.llm_config)
    # instruction = (
    #         '<uploaded_files>\n'
    #         f'/workspace/{workspace_dir_name}\n'
    #         '</uploaded_files>\n'
    #         f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following PR description:\n\n"
    #         f'<pr_description>\n'
    #         f'{instance.problem_statement}\n'
    #         '</pr_description>\n\n'
    #         'Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n'
    #         "I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
    #         'Your task is to make the minimal changes to non-tests files in the /workspace directory to ensure the <pr_description> is satisfied.\n'
    #         '### Follow These Steps to Resolve the Issue:\n'
    # )
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    # Prepare instruction
    if metadata.agent_class == 'CodeActSWEAgent':
        instruction = (
            'We are currently solving the following issue within our repository. Here is the issue text:\n'
            '--- BEGIN ISSUE ---\n'
            f'{instance.problem_statement}\n'
            '--- END ISSUE ---\n\n'
        )
        if USE_HINT_TEXT and instance.hints_text:
            instruction += (
                f'--- BEGIN HINTS ---\n{instance.hints_text}\n--- END HINTS ---\n'
            )
        instruction += CODEACT_SWE_PROMPT.format(workspace_dir_name=workspace_dir_name)
    elif metadata.agent_class == 'CodeActAgent':
        # Instruction based on Anthropic's official trajectory
        # https://github.com/eschluntz/swe-bench-experiments/tree/main/evaluation/verified/20241022_tools_claude-3-5-sonnet-updated/trajs
        instruction = (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n'
            f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following PR description:\n\n"
            f'<pr_description>\n'
            f'{instance.problem_statement}\n'
            '</pr_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            'Your task is to make the minimal changes to non-tests files in the /workspace directory to ensure the <pr_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script to reproduce the error and execute it with `python <filename.py>` using the BashTool, to confirm the error\n'
            '3. Edit the sourcecode of the repo to resolve the issue\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases and make sure your fix handles them as well\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        )
    else:
        # instruction = (
        #     '<uploaded_files>\n'
        #     f'/workspace/{workspace_dir_name}\n'
        #     '</uploaded_files>\n'
        #     f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following PR description:\n\n"
        #     f'<pr_description>\n'
        #     f'{instance.problem_statement}\n'
        #     '</pr_description>\n\n'
        #     'Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n'
        #     "I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        #     'Your task is to make the minimal changes to non-tests files in the /workspace directory to ensure the <pr_description> is satisfied.\n'
        #     'Follow these steps to resolve the issue:\n'
        #     '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
        #     '2. Create a script to reproduce the error and execute it with `python <filename.py>` using the BashTool, to confirm the error\n'
        #     '3. Edit the sourcecode of the repo to resolve the issue\n'
        #     '4. Rerun your reproduce script and confirm that the error is fixed!\n'
        #     '5. Think about edgecases and make sure your fix handles them as well\n'
        #     "Your thinking should be thorough and so it's fine if it's very long.\n"
        # )
        # instruction = (
        #     '<uploaded_files>\n'
        #     f'/workspace/{workspace_dir_name}\n'
        #     '</uploaded_files>\n'
        #     f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following PR description:\n\n"
        #     f'<pr_description>\n'
        #     f'{instance.problem_statement}\n'
        #     '</pr_description>\n\n'
        #     'Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n'
        #     "I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        #     'Your task is to make the minimal changes to non-tests files in the /workspace directory to ensure the <pr_description> is satisfied.\n'
        #     '### Follow These Steps to Resolve the Issue:\n'
        #     '1. **Familiarize Yourself with the Repository**:\n'
        #     '   - Explore the codebase to understand its structure and identify relevant files, classes, functions, or variables that may be affected by the `<pr_description>`.\n'
        #     '2. **Analyze the Problem**:\n'
        #     '   - Identify the specific areas of the codebase that require changes.\n'
        #     '   - Provide a detailed breakdown of the files, code locations, and any related dependencies that need to be addressed.\n'
        #     '3. **Implement the Fix**:\n'
        #     '   - Edit the source code in the identified locations to resolve the issue.\n'
        #     '   - Ensure that your changes are efficient, clean, and adhere to Python best practices.\n'
        #     '4. **Handle Edge Cases**:\n'
        #     '   - Consider potential edge cases and ensure your solution is robust enough to handle them.\n'
        #     '5. **Rerun Your Patch**:\n'
        #     '   - After making the necessary changes, rerun your patch to verify its functionality. Note that you do not need to run any tests yourself; the testing process will be handled by someone else. Once you have completed your changes, simply return it.\n\n'
        #     '### Additional Notes:\n'
        #     '   - Be thorough in your analysis and implementation. It’s okay if your response is detailed and lengthy, as long as it fully addresses the problem.\n'
        #     '   - Clearly document your reasoning, approach, and the changes made to the codebase.\n'
        # )
        # instruction = (
        #     '<uploaded_files>\n'
        #     f'/workspace/{workspace_dir_name}\n'
        #     '</uploaded_files>\n'
        #     f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following PR description:\n\n"
        #     f'<pr_description>\n'
        #     f'{instance.problem_statement}\n'
        #     '</pr_description>\n\n'
        #     'Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n'
        #     "I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        #     'Your task is to make the minimal changes to non-tests files in the /workspace directory to ensure the <pr_description> is satisfied.\n'
        #     '### Follow These Steps to Resolve the Issue:\n'
        #     '1. **Familiarize Yourself with the Repository**:\n'
        #     '   - Explore the codebase to understand its structure and identify relevant files, classes, functions, or variables that may be affected by the `<pr_description>`.\n'
        #     '2. **Analyze the Problem**:\n'
        #     '   - Identify the specific areas of the codebase that require changes.\n'
        #     '   - Provide a detailed breakdown of the files, code locations, and any related dependencies that need to be addressed.\n'
        #     '3. **Implement the Fix**:\n'
        #     '   - Edit the source code in the identified locations to resolve the issue.\n'
        #     '   - Ensure that your changes are efficient, clean, and adhere to Python best practices.\n'
        #     '4. **Handle Edge Cases**:\n'
        #     '   - Consider potential edge cases and ensure your solution is robust enough to handle them.\n'
        #     '5. **Return Your Patch**:\n'
        #     '   - After making the necessary changes, return the patch. Note that you do not need to run any tests yourself; the testing process will be handled by someone else. Once you have completed your changes, simply return it.\n\n'
        #     '### Additional Notes:\n'
        #     '   - Be thorough in your analysis and implementation. It’s okay if your response is detailed and lengthy, as long as it fully addresses the problem.\n'
        #     '   - Clearly document your reasoning, approach, and the changes made to the codebase.\n'
        # )
        # instruction = (
        #     '<uploaded_files>\n'
        #     f'/workspace/{workspace_dir_name}\n'
        #     '</uploaded_files>\n'
        #     f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following PR description:\n\n"
        #     f'<pr_description>\n'
        #     f'{instance.problem_statement}\n'
        #     '</pr_description>\n\n'
        #     'Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n'
        #     "I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        #     'Your task is to make the minimal changes to non-tests files in the /workspace directory to ensure the <pr_description> is satisfied.\n'
        #     '### Follow These Steps to Resolve the Issue:\n'
        #     '1. **Familiarize Yourself with the Repository**:\n'
        #     '   - Explore the codebase to understand its structure and identify relevant files, classes, functions, or variables that may be affected by the `<pr_description>`.\n'
        #     '2. **Analyze the Problem**:\n'
        #     '   - Identify the specific areas of the codebase that require changes.\n'
        #     '   - Provide a detailed breakdown of the files, code locations, and any related dependencies that need to be addressed.\n'
        #     '3. **Implement the Fix**:\n'
        #     '   - Edit the source code in the identified locations to resolve the issue.\n'
        #     '   - Ensure that your changes are efficient, clean, and adhere to Python best practices.\n'
        #     '4. **Handle Edge Cases**:\n'
        #     '   - Consider potential edge cases and ensure your solution is robust enough to handle them.\n'
        #     '5. **Verify Your Patch**:\n'
        #     '   - Create a script to reproduce the error described in the `<pr_description>`.\n'
        #     '   - Do not run the pre-existing test script in the repository, such as `pytest`, but instead run the test script you created with `python <filename.py>` using the BashTool.\n'
        #     '   - Execute the script with `python <filename.py>` using the BashTool to confirm that your changes resolve the issue.\n'
        #     '   - Once the error is fixed, rerun your patch to verify its functionality.\n\n'
        #     '### Additional Notes:\n'
        #     '   - Be thorough in your analysis and implementation. It’s okay if your response is detailed and lengthy, as long as it fully addresses the problem.\n'
        #     '   - Clearly document your reasoning, approach, and the changes made to the codebase.\n'
        # )
        instruction = (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n'
            f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following PR description:\n\n"
            f'<pr_description>\n'
            f'{instance.problem_statement}\n'
            '</pr_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            'Your task is to make the minimal changes to non-tests files in the /workspace directory to ensure the <pr_description> is satisfied.\n'
            '### Follow These Steps to Resolve the Issue:\n'
            '1. **Familiarize Yourself with the Repository**:\n'
            '   - Explore the codebase to understand its structure and identify relevant files, classes, functions, or variables that may be affected by the `<pr_description>`.\n'
            '2. **Analyze the Problem**:\n'
            '   - Identify the specific areas of the codebase that require changes.\n'
            '   - Provide a detailed breakdown of the files, code locations, and any related dependencies that need to be addressed.\n'
            '   - If the PR description provides a script to reproduce the error, extract the script from the PR description to a `/workspace/reproduce_error.py` file.\n'
            '   - Execute the extracted script with `python /workspace/reproduce_error.py` using the BashTool, to confirm the error.\n'
            '   - If the PR description does not provide a script to reproduce the error, do not create a `/workspace/reproduce_error.py` file.\n'
            '3. **Implement the Fix**:\n'
            '   - Edit the source code in the identified locations to resolve the issue.\n'
            '   - Ensure that your changes are efficient, clean, and adhere to Python best practices.\n'
            '4. **Handle Edge Cases**:\n'
            '   - Consider potential edge cases and ensure your solution is robust enough to handle them.\n'
            '5. **Return Your Patch**:\n'
            '   - If the PR description provides a script to reproduce the error, execute the script with `python /workspace/reproduce_error.py` using the BashTool, to confirm the error is fixed.\n'
            '   - If the PR description does not provide a script to reproduce the error, return the patch. Note that you do not need to run any tests yourself; the testing process will be handled by someone else. Once you have completed your changes, simply return it.\n\n'
            '### Additional Notes:\n'
            '   - Be thorough in your analysis and implementation. It’s okay if your response is detailed and lengthy, as long as it fully addresses the problem.\n'
            '   - Clearly document your reasoning, approach, and the changes made to the codebase.\n'
        )

    return instruction

def get_agent_instruction(instance: pd.Series, metadata: EvalMetadata, messages: List[Message], state: State):
    def extract_task_instruction(content: str) -> str:
        """Extracts the content within <task_instruction>...</task_instruction> tags."""
        match = re.search(r'<task_instruction>(.*?)</task_instruction>', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            raise ValueError("Task instruction tags not found in the content.")
    logger.info(f'{"=" * 10} Start Agent Instruction {"=" * 10}')
    logger.info(f'State: {state}')
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    llm = LLM(config=metadata.llm_config)
    first_task = True if len(messages) == 0 else False
    if first_task:
        instruction = (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n'
            f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following PR description:\n\n"
            f'<pr_description>\n'
            f'{instance.problem_statement}\n'
            '</pr_description>\n\n'
            'You are a software engineer tasked with ensuring that the requirements specified in the `<pr_description>` are fully implemented in the repository. However, note the following:\n'
            '1. **You are not responsible for modifying any test files.** All changes to test files mentioned in the `<pr_description>` have already been handled. Therefore, you must **not alter any testing logic or test files** in any way.\n'
            '2. Your task is to make **minimal changes** to the non-test files in the `/workspace` directory to ensure the requirements in the `<pr_description>` are satisfied.\n'
            'However, you are acting as a **team manager** in this scenario. Your responsibilities are as follows:\n'
            '1. **Task Breakdown and Planning:**\n'
            '   - Analyze the requirements in <pr_description> and break them into no more than 5 concise, actionable tasks.\n'
            '   - Create a clear and detailed plan for each task.\n'
            '2. **Task Assignment:** Assign the tasks to individual engineers in your team, ensuring that each task is clearly defined and includes all necessary instructions and requirements.\n'
            '3. **Feedback Integration:** Based on feedback from the assigned engineers, summarize the current status of the tasks and provide a clear next-step action plan.\n'
            '4. **Information Handoff:** Ensure that all relevant and useful information is documented and passed on to the next engineer for seamless collaboration.\n'
            'Now, based on the `<pr_description>`, please create a **task breakdown and planning** for the implementation. Follow these formatting guidelines:\n'
            '1. Write the tasks as a **markdown list**, with each task clearly structured and detailed.\n'
            '2. If the PR description provides a script to reproduce the error, is a good idea to extract the script from the PR description to a `/workspace/reproduce_error.py` file and confirm the error.\n'
            'Now, proceed to create a task breakdown and planning document in the specified format.\n'
        )
        
        messages.append(Message(role='user', content=[TextContent(text=instruction)]))
        logger.info(f'Task Start Instruction: {instruction}')
        response = llm.completion(
            messages=llm.format_messages_for_llm(messages), temperature=1.0, n=1
        )
        
        response = response.choices[0].message.content
        messages.append(Message(role='user', content=[TextContent(text=response)]))
        logger.info(f'Task Start Response: {response}')
        instruction = (
            'Now, generate a detailed and comprehensive **task instruction** for the first task, ensuring it includes all necessary requirements for the assigned engineer to proceed efficiently. Follow these guidelines when creating the task instruction:\n'
            '1. **Do not repeat** information already provided in `<uploaded_files>` and `<pr_description>`, as this content will automatically be appended to the beginning of the task instruction.\n'
            '2. The **task instruction** should be specifically intended for another engineer to implement, so ensure it is clear, actionable, and precise.\n'
            '3. Include all relevant details, steps, and context required to complete the task without ambiguity or unnecessary back-and-forth communication.\n'
            '4. Wrap the **task instruction** in `<task_instruction>...</task_instruction>` tags to ensure it can be properly parsed by the system.\n'
            '5. Use a conversational and approachable style to make the instructions easy to follow. For example, use phrases like "You are..." and "Your task is..." to directly address the engineer.\n'
            f'6. If reproducing an error is required, include instructions to write a `reproduce_error.py` file in the `/workspace/{workspace_dir_name}` directory. Use the `BashTool` to run the script for error reproduction.\n'
            '7. Avoid using `pdb` or similar debugging tools in the task.\n'
            'Remember, the goal is to provide the engineer with everything they need to start and complete the task efficiently.\n'
        )
        # do not use pdb here
        # if you want to reproduce the error, please write the reproduce_error.py file in the /workspace/{workspace_dir_name} directory, and use the BashTool to run the script.
    else:
        instruction = (
            'The following is the what the previous engineer has done:\n'
            '<progress_of_previous_engineer>\n'
            f'{state.outputs["thought"]}\n'
            '</progress_of_previous_engineer>\n'
            'Now, generate a detailed and comprehensive **task instruction** for the next task, ensuring it includes all necessary requirements for the assigned engineer to proceed efficiently. Follow these guidelines when creating the task instruction:\n'
            '1. **Do not repeat** information already provided in `<uploaded_files>` and `<pr_description>`, as this content will automatically be appended to the beginning of the task instruction.\n'
            '2. The **task instruction** should be specifically intended for another engineer to implement, so ensure it is clear, actionable, and precise.\n'
            '3. Include all relevant details, steps, and context required to complete the task without ambiguity or unnecessary back-and-forth communication.\n'
            '4. Wrap the **task instruction** in `<task_instruction>...</task_instruction>` tags to ensure it can be properly parsed by the system.\n'
            '5. Use a conversational and approachable style to make the instructions easy to follow. For example, use phrases like "You are..." and "Your task is..." to directly address the engineer.\n'
            f'6. If reproducing an error is required, include instructions to write a `reproduce_error.py` file in the `/workspace/{workspace_dir_name}` directory. Use the `BashTool` to run the script for error reproduction.\n'
            '7. Avoid using `pdb` or similar debugging tools in the task.\n'
            'Remember, the goal is to provide the engineer with everything they need to start and complete the task efficiently.\n'
        )
        
        check_finish_instruction = (
            'The following is the what the previous engineer has done:\n'
            '<progress_of_previous_engineer>\n'
            f'{state.outputs["thought"]}\n'
            '</progress_of_previous_engineer>\n'
            'Please first analyze the progress of the previous engineer.\n'
            'Then, determine if the sub-task is completed. If it is, please return [[SUBTASK_FINISH]] in the response. Otherwise, return [[SUBTASK_CONTINUE]].\n'
            'Finally, determine if all tasks are completed. If it is, please return [[ALLTASKS_FINISH]] in the response. Otherwise, return [[ALLTASKS_CONTINUE]].'
        )
        logger.info(f'Check Finish Instruction: {check_finish_instruction}')
        check_finish_response = llm.completion(
            messages=llm.format_messages_for_llm(messages + [Message(role='user', content=[TextContent(text=check_finish_instruction)])]), temperature=1.0, n=1
        )
        check_finish_response = check_finish_response.choices[0].message.content
        logger.info(f'Check Finish Response: {check_finish_response}')
        if '[[ALLTASKS_FINISH]]' in check_finish_response:
            return messages, '[[ALLTASKS_FINISH]]', True

    messages.append(Message(role='user', content=[TextContent(text=instruction)]))
    logger.info(f'Task Instruction: {instruction}')
    while True:
        try:
            response = llm.completion(
                messages=llm.format_messages_for_llm(messages), temperature=1.0, n=1
            )
            response = response.choices[0].message.content
            logger.info(f'Task Response: {response}')
            task_instruction = extract_task_instruction(response)
            break
        except Exception as e:
            logger.error(f'Failed to extract task instruction: {e}')
            continue
    if first_task:
        final_task_instruction = (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n\n'
            f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following PR description:\n\n"
            f'<pr_description>\n'
            f'{instance.problem_statement}\n'
            '</pr_description>\n\n'
            f'{task_instruction}\n'
        )
    else:
        final_task_instruction = (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n\n'
            f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following PR description:\n\n"
            f'<pr_description>\n'
            f'{instance.problem_statement}\n'
            '</pr_description>\n\n'
            'The following is the what the previous engineer has done:\n'
            '<progress_of_previous_engineer>\n'
            f'{state.outputs["thought"]}\n'
            '</progress_of_previous_engineer>\n\n'
            f'{task_instruction}\n'
        )
        
    logger.info(f'{"=" * 10} End Agent Instruction {"=" * 10}')
    return messages, final_task_instruction, False

# TODO: migrate all swe-bench docker to ghcr.io/openhands
DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')


def get_instance_docker_image(instance_id: str) -> str:
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> AppConfig:
    SWE_BENCH_CONTAINER_IMAGE = 'ghcr.io/opendevin/eval-swe-bench:full-v1.2.1'
    if USE_INSTANCE_IMAGE:
        # We use a different instance image for the each instance of swe-bench eval
        base_container_image = get_instance_docker_image(instance['instance_id'])
        logger.info(
            f'Using instance container image: {base_container_image}. '
            f'Please make sure this image exists. '
            f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
        )
    else:
        base_container_image = SWE_BENCH_CONTAINER_IMAGE
        logger.info(f'Using swe-bench container image: {base_container_image}')

    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        max_budget_per_task=5.0,
        runtime=os.environ.get('RUNTIME', 'eventstream'),
        sandbox=SandboxConfig(
            base_container_image=base_container_image,
            enable_auto_lint=True,
            use_host_network=False,
            # large enough timeout, since some testcases take very long to run
            timeout=300,
            # Add platform to the sandbox config to solve issue 4401
            platform='linux/amd64',
            api_key=os.environ.get('ALLHANDS_API_KEY', None),
            remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL'),
            keep_remote_runtime_alive=False,
        ),
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    if metadata.llm_config.log_completions:
        metadata.llm_config.log_completions_folder = os.path.join(
            metadata.eval_output_dir, 'llm_completions', instance['instance_id']
        )
        logger.info(
            f'Logging LLM completions for instance {instance["instance_id"]} to '
            f'{metadata.llm_config.log_completions_folder}'
        )
    config.set_llm_config(metadata.llm_config)
    agent_config = AgentConfig(
        codeact_enable_jupyter=False,
        codeact_enable_browsing_delegate=False,
        codeact_enable_llm_editor=False,
    )
    config.set_agent_config(agent_config)
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    instance_base_commit = instance['base_commit']
    instance_id = instance['instance_id']
    obs: CmdOutputObservation
    # Set instance id
    action = CmdRunAction(
        command=f"""echo 'export SWE_INSTANCE_COMMIT={instance_base_commit}' >> ~/.bashrc && echo 'export SWEBENCH_WORKSPACE=/workspace/{workspace_dir_name}' >> ~/.bashrc && echo 'export SWE_INSTANCE_ID={instance_id}' >> ~/.bashrc && echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && echo "alias git='git --no-pager'" >> ~/.bashrc"""
    )
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0, f'Failed to export SWE_INSTANCE_ID: {str(obs)}'
    )

    action = CmdRunAction(command="""export USER=$(whoami); echo USER=${USER} """)
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to export USER: {str(obs)}')

    if USE_INSTANCE_IMAGE:
        # inject the init script
        script_dir = os.path.dirname(__file__)

        # inject the instance info
        action = CmdRunAction(command='mkdir -p /swe_util/eval_data/instances')
        action.timeout = 600
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to create /swe_util/eval_data/instances: {str(obs)}',
        )

        swe_instance_json_name = 'swe-bench-instance.json'
        with tempfile.TemporaryDirectory() as temp_dir:
            # Construct the full path for the desired file name within the temporary directory
            temp_file_path = os.path.join(temp_dir, swe_instance_json_name)
            # Write to the file with the desired name within the temporary directory
            with open(temp_file_path, 'w') as f:
                if not isinstance(instance, dict):
                    json.dump([instance.to_dict()], f)
                else:
                    json.dump([instance], f)

            # Copy the file to the desired location
            runtime.copy_to(temp_file_path, '/swe_util/eval_data/instances/')

        # inject the instance swe entry
        runtime.copy_to(
            str(os.path.join(script_dir, 'scripts/setup/instance_swe_entry.sh')),
            '/swe_util/',
        )
        action = CmdRunAction(command='cat ~/.bashrc')
        action.timeout = 600
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(obs.exit_code == 0, f'Failed to cat ~/.bashrc: {str(obs)}')

        action = CmdRunAction(command='source ~/.bashrc')
        action.timeout = 600
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if isinstance(obs, ErrorObservation):
            logger.error(f'Failed to source ~/.bashrc: {str(obs)}')
        assert_and_raise(obs.exit_code == 0, f'Failed to source ~/.bashrc: {str(obs)}')

        action = CmdRunAction(command='source /swe_util/instance_swe_entry.sh')
        action.timeout = 3600
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to source /swe_util/instance_swe_entry.sh: {str(obs)}',
        )

        # action = CmdRunAction(command='cat /openhands/code/openhands/runtime/plugins/agent_skills/file_editor/impl.py')
        # action = IPythonRunCellAction(code='import openhands; import os; print(os.path.dirname(openhands.__file__))')
        # action = IPythonRunCellAction(code='print(file_editor("view", path="/swe_util/instance_swe_entry.sh"))')
        # action = IPythonRunCellAction(code='print(file_editor("str_replace", path="/openhands/code/openhands/runtime/plugins/agent_skills/file_editor/test.py", old_str="insert_line=65))", new_str="print( ;;;"))')
        # action.timeout = 600
        # logger.info(action, extra={'msg_type': 'ACTION'})
        # obs = runtime.run_action(action)
        # logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        # exit(0)
    else:
        action = CmdRunAction(command='source /swe_util/swe_entry.sh')
        action.timeout = 1800
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to source /swe_util/swe_entry.sh: {str(obs)}',
        )

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )

    action = CmdRunAction(command='git reset --hard')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to git reset --hard: {str(obs)}')

    action = CmdRunAction(
        command='for remote_name in $(git remote); do git remote remove "${remote_name}"; done'
    )
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to remove git remotes: {str(obs)}')

    logger.info('-' * 30)
    logger.info('END Runtime Initialization Fn')
    logger.info('-' * 30)


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required, but it is used to get the workspace_dir_name
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )

    action = CmdRunAction(command='git config --global core.pager ""')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git config --global core.pager "": {str(obs)}',
    )

    action = CmdRunAction(command='git add -A')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git add -A: {str(obs)}',
    )

    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(
            command=f'git diff --no-color --cached {instance["base_commit"]}',
            keep_prompt=False,
        )
        action.timeout = 600 + 100 * n_retries
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        n_retries += 1
        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                git_patch = obs.content.strip()
                break
            else:
                logger.info('Failed to get git diff, retrying...')
                sleep_if_should_continue(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error occurred: {obs.content}. Retrying...')
            sleep_if_should_continue(10)
        else:
            assert_and_raise(False, f'Unexpected observation type: {str(obs)}')

    assert_and_raise(git_patch is not None, 'Failed to get git diff (None)')

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)
    return {'git_patch': git_patch}


def load_docker_image(base_container_image: str):
    def load_docker_image_from_path(image_path):
        try:
            logger.info(f'Loading docker image: {image_path}')
            docker_client = docker.from_env(timeout=600)

            # 读取文件的二进制内容
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            # 加载镜像
            docker_client.images.load(image_data)
            logger.info('Docker image loaded successfully.')

        except docker.errors.ImageLoadError as e:
            logger.error(f'Failed to load image: {e}')
        except Exception as e:
            logger.error(f'An unexpected error occurred: {e}')

    def image_exists(image_name: str) -> bool:
        """Check if the image exists in the local store.

        Args:
            image_name (str): The Docker image to check (<image repo>:<image tag>)
        Returns:
            bool: Whether the Docker image exists in the registry or in the local store
        """
        docker_client = docker.from_env()
        if not image_name:
            logger.error(f'Invalid image name: `{image_name}`')
            return False

        try:
            logger.debug(f'Checking, if image exists locally:\n{image_name}')
            docker_client.images.get(image_name)
            logger.debug('Image found locally.')
            return True
        except docker.errors.ImageNotFound:
            logger.debug(f'Image {image_name} not found locally')
            return False

    # import pdb; pdb.set_trace()
    runtime_image_repo, _ = get_runtime_image_repo_and_tag(base_container_image)
    lock_tag = f'oh_v{oh_version}_{get_hash_for_lock_files(base_container_image)}'
    image_path = f'{DOCKER_IMAGE_DIR}/{runtime_image_repo}:{lock_tag}.tar'
    if image_exists(f'{runtime_image_repo}:{lock_tag}'):
        logger.info(f'Docker image {runtime_image_repo}:{lock_tag} already exists')
    elif os.path.exists(image_path):
        # lock_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'docker_image_load.lock')
        # with FileLock(lock_file_path):
        load_docker_image_from_path(image_path)
    else:
        logger.info(f'Docker image not found: {image_path}')


def remove_docker_image(base_container_image: str):
    # import pdb; pdb.set_trace()
    runtime_image_repo, _ = get_runtime_image_repo_and_tag(base_container_image)
    lock_tag = f'oh_v{oh_version}_{get_hash_for_lock_files(base_container_image)}'
    source_tag = f'{lock_tag}_{get_hash_for_source_files()}'
    docker_client = docker.from_env(timeout=120)
    lock_image_name = f'{runtime_image_repo}:{lock_tag}'
    source_image_name = f'{runtime_image_repo}:{source_tag}'
    image_path = f'{DOCKER_IMAGE_DIR}/{runtime_image_repo}:{lock_tag}.tar'
    logger.info(f'Removing docker image: {source_image_name}')
    docker_client.images.remove(source_image_name, force=True)
    if os.path.exists(image_path):
        logger.info(f'Removing docker image: {lock_image_name}')
        docker_client.images.remove(lock_image_name, force=True)
    else:
        logger.info(f'Docker image {lock_image_name} will be kept.')


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    time.sleep(random.randint(1, 10))
    
    config = get_config(instance, metadata)
    # print(metadata.llm_config.api_key)

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    # dockerpull.org/xingyaoww/sweb.eval.x86_64.django_s_django-15781
    load_docker_image(config.sandbox.base_container_image)
    runtime = create_runtime(config, sid=f'{instance.instance_id}_{current_time}')
    call_async_from_sync(runtime.connect)

    max_steps = 5
    try:
        initialize_runtime(runtime, instance)
        print(runtime.api_url)
        # import pdb; pdb.set_trace()
        state = None
        messages = []
        for i_step in range(max_steps):
            # import pdb; pdb.set_trace()
            messages, instruction, finished = get_agent_instruction(instance, metadata, messages, state)
            if finished:
                logger.info(f'Task {instance.instance_id} finished.')
                break
            # Here's how you can run the agent (similar to the `main` function) and get the final task state
            # runtime.event_stream.clear()
            state: State | None = asyncio.run(
                run_controller(
                    config=config,
                    initial_user_action=MessageAction(content=instruction),
                    runtime=runtime,
                    fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[
                        metadata.agent_class
                    ],
                )
            )
            logger.info(f'State: {state}')

            # if fatal error, throw EvalError to trigger re-run
            if (
                state.last_error
                and 'fatal error during agent execution' in state.last_error
                and 'stuck in a loop' not in state.last_error
            ):
                raise EvalException('Fatal error detected: ' + state.last_error)

        # ======= THIS IS SWE-Bench specific =======
        # Get git patch
        return_val = complete_runtime(runtime, instance)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )
    finally:
        runtime.close(rm_all_containers=False)
    # ==========================================

    # ======= Attempt to evaluate the agent's edits =======
    # we use eval_infer.sh to evaluate the agent's edits, not here
    # because the agent may alter the environment / testcases
    test_result = {
        'git_patch': git_patch,
    }

    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError('State should not be None.')

    histories = [event_to_dict(event) for event in state.history.get_events()]
    metrics = state.metrics.get() if state.metrics else None

    # Save the output
    output = EvalOutput(
        instance_id=instance.instance_id,
        instruction=instruction,
        instance=instance.to_dict(),  # SWE Bench specific
        test_result=test_result,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )

    remove_docker_image(config.sandbox.base_container_image)
    return output


def filter_dataset(dataset: pd.DataFrame, filter_column: str) -> pd.DataFrame:
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = toml.load(file)
            if 'selected_ids' in data:
                selected_ids = data['selected_ids']
                logger.info(
                    f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                )
                subset = dataset[dataset[filter_column].isin(selected_ids)]
                logger.info(f'Retained {subset.shape[0]} tasks after filtering')
                return subset
    return dataset


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
    # import pdb; pdb.set_trace()
    args, _ = parser.parse_known_args()

    # NOTE: It is preferable to load datasets from huggingface datasets and perform post-processing
    # so we don't need to manage file uploading to OpenHands's repo
    dataset = load_dataset(args.dataset, split=args.split)
    logger.info(f'Loaded dataset {args.dataset} with split {args.split}')
    swe_bench_tests = filter_dataset(dataset.to_pandas(), 'instance_id')

    llm_config = None
    # import pdb; pdb.set_trace()
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        llm_config.log_completions = True

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    details = {}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    dataset_descrption = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )
    metadata = make_metadata(
        llm_config,
        dataset_descrption,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    instances = prepare_dataset(swe_bench_tests, output_file, args.eval_n_limit)

    if len(instances) > 0 and not isinstance(
        instances['PASS_TO_PASS'][instances['PASS_TO_PASS'].index[0]], str
    ):
        for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
            instances[col] = instances[col].apply(lambda x: str(x))

    run_evaluation(
        instances, metadata, output_file, args.eval_num_workers, process_instance
    )
