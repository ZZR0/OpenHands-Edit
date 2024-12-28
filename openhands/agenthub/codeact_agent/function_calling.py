"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json
import warnings
from typing import Any

import numpy as np
from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ModelResponse,
)
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from swebench.harness.test_spec import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.utils import get_test_directives

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentDelegateAction,
    AgentFinishAction,
    CmdRunAction,
    FileEditAction,
    IPythonRunCellAction,
    MessageAction,
    RunRegressionAction,
)
from openhands.events.tool import ToolCallMetadata

# 忽略特定的警告
warnings.filterwarnings('ignore', category=ConvergenceWarning)

SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.
</IMPORTANT>
"""

_BASH_DESCRIPTION = """Execute a bash command in the terminal.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
"""

CmdRunTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='execute_bash',
        description=_BASH_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)

_IPYTHON_DESCRIPTION = """Run a cell of Python code in an IPython environment.
* The assistant should define variables and import packages before using them.
* The variable defined in the IPython environment will not be available outside the IPython environment (e.g., in terminal).
"""
# We are not using agentskills's file_ops for viewing files now because StrReplaceEditorTool already supports viewing files
# """* Apart from the standard Python library, the assistant can also use the following functions (already imported):
# {AgentSkillsRequirement.documentation}"""

IPythonTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='execute_ipython_cell',
        description=_IPYTHON_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'code': {
                    'type': 'string',
                    'description': 'The Python code to execute. Supports magic commands like %pip.',
                },
            },
            'required': ['code'],
        },
    ),
)

_FILE_EDIT_DESCRIPTION = """Edit a file.
* The assistant can edit files by specifying the file path and providing a draft of the new file content.
* The draft content doesn't need to be exactly the same as the existing file; the assistant may skip unchanged lines using comments like `# unchanged` to indicate unchanged sections.
* IMPORTANT: For large files (e.g., > 300 lines), specify the range of lines to edit using `start` and `end` (1-indexed, inclusive). The range should be smaller than 300 lines.
* To append to a file, set both `start` and `end` to `-1`.
* If the file doesn't exist, a new file will be created with the provided content.

**Example 1: general edit for short files**
For example, given an existing file `/path/to/file.py` that looks like this:
(this is the end of the file)
1|class MyClass:
2|    def __init__(self):
3|        self.x = 1
4|        self.y = 2
5|        self.z = 3
6|
7|print(MyClass().z)
8|print(MyClass().x)
(this is the end of the file)

The assistant wants to edit the file to look like this:
(this is the end of the file)
1|class MyClass:
2|    def __init__(self):
3|        self.x = 1
4|        self.y = 2
5|
6|print(MyClass().y)
(this is the end of the file)

The assistant may produce an edit action like this:
path="/path/to/file.txt" start=1 end=-1
content=```
class MyClass:
    def __init__(self):
        # no changes before
        self.y = 2
        # self.z is removed

# MyClass().z is removed
print(MyClass().y)
```

**Example 2: append to file for short files**
For example, given an existing file `/path/to/file.py` that looks like this:
(this is the end of the file)
1|class MyClass:
2|    def __init__(self):
3|        self.x = 1
4|        self.y = 2
5|        self.z = 3
6|
7|print(MyClass().z)
8|print(MyClass().x)
(this is the end of the file)

To append the following lines to the file:
```python
print(MyClass().y)
```

The assistant may produce an edit action like this:
path="/path/to/file.txt" start=-1 end=-1
content=```
print(MyClass().y)
```

**Example 3: edit for long files**

Given an existing file `/path/to/file.py` that looks like this:
(1000 more lines above)
1001|class MyClass:
1002|    def __init__(self):
1003|        self.x = 1
1004|        self.y = 2
1005|        self.z = 3
1006|
1007|print(MyClass().z)
1008|print(MyClass().x)
(2000 more lines below)

The assistant wants to edit the file to look like this:

(1000 more lines above)
1001|class MyClass:
1002|    def __init__(self):
1003|        self.x = 1
1004|        self.y = 2
1005|
1006|print(MyClass().y)
(2000 more lines below)

The assistant may produce an edit action like this:
path="/path/to/file.txt" start=1001 end=1008
content=```
class MyClass:
    def __init__(self):
        # no changes before
        self.y = 2
        # self.z is removed

# MyClass().z is removed
print(MyClass().y)
```
"""

LLMBasedFileEditTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='edit_file',
        description=_FILE_EDIT_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'The absolute path to the file to be edited.',
                },
                'new_content_draft': {
                    'type': 'string',
                    'description': 'A draft of the new content for the file being edited. Note that the assistant may skip unchanged lines.',
                },
                'start': {
                    'type': 'integer',
                    'description': 'The starting line number for the edit (1-indexed, inclusive). Default is 1.',
                },
                'end': {
                    'type': 'integer',
                    'description': 'The ending line number for the edit (1-indexed, inclusive). Default is -1 (end of file).',
                },
            },
            'required': ['path', 'content'],
        },
    ),
)

_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

StrReplaceEditorTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='str_replace_editor',
        description=_STR_REPLACE_EDITOR_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'description': 'The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.',
                    'enum': ['view', 'create', 'str_replace', 'insert', 'undo_edit'],
                    'type': 'string',
                },
                'path': {
                    'description': 'Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.',
                    'type': 'string',
                },
                'file_text': {
                    'description': 'Required parameter of `create` command, with the content of the file to be created.',
                    'type': 'string',
                },
                'old_str': {
                    'description': 'Required parameter of `str_replace` command containing the string in `path` to replace.',
                    'type': 'string',
                },
                'new_str': {
                    'description': 'Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.',
                    'type': 'string',
                },
                'insert_line': {
                    'description': 'Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.',
                    'type': 'integer',
                },
                'view_range': {
                    'description': 'Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.',
                    'items': {'type': 'integer'},
                    'type': 'array',
                },
            },
            'required': ['command', 'path'],
        },
    ),
)

_BROWSER_DELEGATION = """Delegate the task to another browsing agent.
The assistant should delegate the task if it needs to browse the Internet.
"""

BrowserDelegationTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='delegate_to_browsing_agent',
        description=_BROWSER_DELEGATION,
        parameters={
            'type': 'object',
            'properties': {
                'task': {
                    'type': 'string',
                    'description': 'The task for the browsing agent to execute. It should include all the necessary context and specify what information the browsing agent should return.',
                },
            },
            'required': ['task'],
        },
    ),
)

_REGRESSION_DESCRIPTION = """Run regression tests if you think you have made a preliminary change that can resolve this issue.
"""

RegressionTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='run_regression',
        description=_REGRESSION_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {},
            'required': [],
        },
    ),
)

_FINISH_DESCRIPTION = """Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task."""

FinishTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='finish',
        description=_FINISH_DESCRIPTION,
    ),
)


def combine_thought(action: Action, thought: str) -> Action:
    if not hasattr(action, 'thought'):
        return action
    if thought:
        action.thought = thought
    return action


def get_all_keys(d):
    """递归获取字典中的所有键值"""
    keys = []
    for key, value in d.items():
        keys.append(key)
        if isinstance(value, dict):
            keys.extend(get_all_keys(value))
    return keys


def get_all_key_values(d):
    """递归获取字典中的所有键值对

    Args:
        d: 要处理的字典

    Returns:
        list: 包含所有键值对的列表，格式为 [(key_path, value), ...]
    """
    items = []
    for key, value in d.items():
        if isinstance(value, dict):
            items.extend(get_all_key_values(value))
        items.append((key, value))
    return items


def response_to_actions(
    response: ModelResponse, instance: dict[str, Any] | None = None
) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1, 'Only one choice is supported for now'
    assistant_msg = response.choices[0].message
    if assistant_msg.tool_calls:
        # Check if there's assistant_msg.content. If so, add it to the thought
        thought = ''
        if isinstance(assistant_msg.content, str):
            thought = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if msg['type'] == 'text':
                    thought += msg['text']

        # Process each tool call to OpenHands action
        for i, tool_call in enumerate(assistant_msg.tool_calls):
            action: Action
            try:
                raw_arguments = json.loads(tool_call.function.arguments)
                logger.info(f'Raw arguments: {raw_arguments}')
                # {'fields': {'value': {'string_value': 'view', 'path': '/workspace/astropy__astropy__5.1/astropy/io/ascii/rst.py'}, 'key': 'command'}}
                # {'fields': {'value': {'string_value': '/workspace/astropy__astropy__5.2/astropy/nddata/mixins/ndarithmetic.py'}, 'key': 'path'}}
                arguments = {}
                all_keys = get_all_keys(raw_arguments)
                if 'fields' in all_keys:
                    all_key_values = get_all_key_values(raw_arguments)
                    _arguments = {k: v for k, v in all_key_values}
                    for key in _arguments:
                        if isinstance(_arguments[key], dict):
                            for v_key in _arguments[key]:
                                if 'value' in v_key:
                                    break
                            _arguments[key] = _arguments[key][v_key]

                    if 'key' in _arguments and 'value' in _arguments:
                        _arguments[_arguments['key']] = _arguments['value']
                    for key in _arguments:
                        if 'key' in key or 'value' in key or 'fields' in key:
                            continue
                        arguments[key] = _arguments[key]
                else:
                    arguments = raw_arguments
                logger.info(f'New arguments: {arguments}')
                tool_call.function.arguments = json.dumps(arguments)
            except Exception as e:
                raise RuntimeError(
                    f'Failed to parse tool call arguments: {tool_call.function.arguments}'
                ) from e
            if tool_call.function.name == 'execute_bash':
                action = CmdRunAction(**arguments)
            elif tool_call.function.name == 'execute_ipython_cell':
                action = IPythonRunCellAction(**arguments)
            elif tool_call.function.name == 'delegate_to_browsing_agent':
                action = AgentDelegateAction(
                    agent='BrowsingAgent',
                    inputs=arguments,
                )
            elif tool_call.function.name == 'finish':
                action = AgentFinishAction()
            elif tool_call.function.name == 'edit_file':
                action = FileEditAction(**arguments)
            elif tool_call.function.name == 'str_replace_editor':
                # We implement this in agent_skills, which can be used via Jupyter
                # convert tool_call.function.arguments to kwargs that can be passed to file_editor
                code = f'print(file_editor(**{arguments}))'
                logger.debug(
                    f'TOOL CALL: str_replace_editor -> file_editor with code: {code}'
                )
                action = IPythonRunCellAction(code=code, include_extra=False)
            elif tool_call.function.name == 'run_regression':
                # raise NotImplementedError('Regression tool is not implemented yet')

                assert (
                    instance
                ), 'Instance metadata must be set if regression is enabled.'
                instance_id = instance['instance_id']

                # Convert e.g. "logs/scikit-learn__scikit-learn-12421/test_output.txt" to "scikit-learn/scikit-learn"
                repo = '-'.join(
                    instance_id.replace('__', '/').split('-')[:-1]
                )  # e.g. scikit-learn/scikit-learn

                test_command = ' '.join(
                    [
                        MAP_REPO_VERSION_TO_SPECS[instance['repo']][
                            instance['version']
                        ]['test_cmd'],
                        *get_test_directives(instance),
                    ]
                )
                if 'pytest ' in test_command:
                    test_command = test_command.replace('-rA ', '-rA -s ')

                action = RunRegressionAction(
                    repo=repo,
                    version=instance['version'],
                    test_command=test_command,
                    testcases=instance['initial_passed_tests'],
                )
            else:
                raise RuntimeError(f'Unknown tool call: {tool_call.function.name}')

            # We only add thought to the first action
            if i == 0:
                action = combine_thought(action, thought)
            # Add metadata for tool calling
            action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=tool_call.id,
                function_name=tool_call.function.name,
                model_response=response,
                total_calls_in_response=len(assistant_msg.tool_calls),
            )
            actions.append(action)
    else:
        actions.append(
            MessageAction(content=assistant_msg.content, wait_for_response=True)
        )

    assert len(actions) >= 1
    return actions


def get_tools(
    codeact_enable_browsing_delegate: bool = False,
    codeact_enable_llm_editor: bool = False,
    codeact_enable_jupyter: bool = False,
    codeact_enable_regression: bool = False,
) -> list[ChatCompletionToolParam]:
    tools = [CmdRunTool, FinishTool]
    if codeact_enable_regression:
        tools.append(RegressionTool)
    if codeact_enable_browsing_delegate:
        tools.append(BrowserDelegationTool)
    if codeact_enable_jupyter:
        tools.append(IPythonTool)
    if codeact_enable_llm_editor:
        tools.append(LLMBasedFileEditTool)
    else:
        tools.append(StrReplaceEditorTool)
    return tools


def select_by_kmeans(embeddings: list[float]) -> tuple[int, list[int]]:
    # 自动选择最优簇数并进行聚类
    # 将嵌入转换为数组
    embedding_array = np.array(embeddings)

    # 初始化变量
    max_k = min(len(embedding_array) - 1, 10)  # 最大簇数
    best_k = 2
    best_score = -1

    # 计算每个簇数的轮廓系数
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embedding_array)
        score = (
            silhouette_score(embedding_array, labels)
            if len(np.unique(labels)) > 1
            else 0
        )

        if score > best_score:
            best_score = score
            best_k = k

    # 使用最佳簇数重新进行K-means聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(embedding_array)
    # 找到最大簇
    unique, counts = np.unique(labels, return_counts=True)
    max_cluster_label = unique[np.argmax(counts)]

    # 找到最大簇的类中心
    max_cluster_center = kmeans.cluster_centers_[max_cluster_label]

    # 计算最大簇中每个点到类中心的距离
    max_cluster_indices = [
        i for i, label in enumerate(labels) if label == max_cluster_label
    ]
    distances = [
        np.linalg.norm(np.array(embeddings[i]) - max_cluster_center)
        for i in max_cluster_indices
    ]

    # 找到距离类中心最近的选项
    closest_index = max_cluster_indices[np.argmin(distances)]

    return closest_index, labels


def select_by_dbscan(embeddings: list[float]) -> tuple[int, list[int]]:
    # 将嵌入转换为数组
    embedding_array = np.array(embeddings)

    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=0.5, min_samples=2)  # 可以根据数据调整eps和min_samples
    labels = dbscan.fit_predict(embedding_array)

    # 找到最大簇
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    max_cluster_label = unique[np.argmax(counts)]

    # 找到最大簇的类中心
    max_cluster_indices = [
        i for i, label in enumerate(labels) if label == max_cluster_label
    ]
    max_cluster_center = np.mean([embeddings[i] for i in max_cluster_indices], axis=0)

    # 计算最大簇中每个点到类中心的距离
    distances = [
        np.linalg.norm(np.array(embeddings[i]) - max_cluster_center)
        for i in max_cluster_indices
    ]

    # 找到距离类中心最近的选项
    closest_index = max_cluster_indices[np.argmin(distances)]
    return closest_index, labels


def select_by_optics(embeddings: list[float]) -> tuple[int, list[int]]:
    # 使用OPTICS进行聚类
    # 将嵌入转换为数组
    embedding_array = np.array(embeddings)

    # 使用OPTICS进行聚类
    optics = OPTICS(min_samples=2)  # 可以根据数据调整min_samples
    labels = optics.fit_predict(embedding_array)
    print(labels)
    # 找到最大簇
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    max_cluster_label = unique[np.argmax(counts)]

    # 找到最大簇的类中心
    max_cluster_indices = [
        i for i, label in enumerate(labels) if label == max_cluster_label
    ]
    max_cluster_center = np.mean([embeddings[i] for i in max_cluster_indices], axis=0)

    # 计算最大簇中每个点到类中心的距离
    distances = [
        np.linalg.norm(np.array(embeddings[i]) - max_cluster_center)
        for i in max_cluster_indices
    ]

    # 找到距离类中心最近的选项
    closest_index = max_cluster_indices[np.argmin(distances)]

    return closest_index, labels
