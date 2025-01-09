import re
from typing import Any

from swebench.harness.test_spec import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.utils import get_test_directives

from openhands.controller.action_parser import (
    ActionParser,
    ResponseParser,
)
from openhands.core.exceptions import LLMMalformedActionError
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


class CodeActResponseParser(ResponseParser):
    """Parser action:
    - CmdRunAction(command) - bash command to run
    - FileEditAction(path, content) - edit a file
    - IPythonRunCellAction(code) - IPython code to run
    - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
    - MessageAction(content) - Message action to run (e.g. ask for clarification)
    - AgentFinishAction() - end the interaction
    """

    def __init__(self, instance: dict[str, Any] | None = None):
        # Need pay attention to the item order in self.action_parsers
        super().__init__()
        self.instance = instance
        self.action_parsers = [
            CodeActActionParserFinish(),
            CodeActActionParserCmdRun(),
            CodeActActionParserIPythonRunCell(),
            CodeActActionParserFileEdit(),
            CodeActActionParserRunRegression(instance),
            CodeActActionParserAgentDelegate(),
        ]
        self.default_parser = CodeActActionParserMessage()

    def parse(self, response) -> Action:
        action_str = self.parse_response(response)
        return self.parse_action(action_str)

    def parse_response(self, response) -> str:
        MAP_LANG_TO_COMMAND = {
            'bash': 'execute_bash',
            'ipython': 'execute_ipython',
            'browse': 'execute_browse',
            'regression': 'run_regression',
        }

        action = response.choices[0].message.content
        if action is None:
            return ''
        for lang in ['bash', 'ipython', 'browse', 'regression']:
            command = MAP_LANG_TO_COMMAND[lang]
            # special handling for DeepSeek: it has stop-word bug and returns </execute_ipython instead of </execute_ipython>
            if f'</{command}' in action and f'</{command}>' not in action:
                action = action.replace(f'</{command}', f'</{command}>')
            # special handling for Gemini: it has stop-word bug and returns </execute_ipython></file_edit> instead of </execute_ipython>
            # if (
            #     f'<execute_{lang}>' in action
            #     and f'</execute_{lang}></file_edit>' in action
            # ):
            #     action = action.replace(
            #         f'</execute_{lang}></file_edit>', f'</execute_{lang}>'
            #     )
            # if f'<execute_{lang}>' in action and f'</execute_{lang}>' not in action:
            #     action += f'</execute_{lang}>'

        # if (
        #     '<file_edit' in action
        #     and '</file_edit>' not in action
        #     and not any(
        #         f'</execute_{lang}>' in action for lang in ['bash', 'ipython', 'browse']
        #     )
        # ):
        #     action += '</file_edit>'
        return action

    def parse_action(self, action_str: str) -> Action:
        instance = self.instance
        for action_parser in self.action_parsers:
            if (
                isinstance(action_parser, CodeActActionParserRunRegression)
                and not instance
            ):
                continue
            if action_parser.check_condition(action_str):
                return action_parser.parse(action_str)
        return self.default_parser.parse(action_str)

    def action_to_str(self, action: Action) -> str:
        if isinstance(action, CmdRunAction):
            return (
                f'{action.thought}\n<execute_bash>\n{action.command}\n</execute_bash>'
            )
        elif isinstance(action, IPythonRunCellAction):
            return f'{action.thought}\n<execute_ipython>\n{action.code}\n</execute_ipython>'
        elif isinstance(action, AgentDelegateAction):
            return f'{action.thought}\n<execute_browse>\n{action.inputs["task"]}\n</execute_browse>'
        elif isinstance(action, FileEditAction):
            return f'{action.thought}\n<file_edit path={action.path}>\n{action.content}\n</file_edit>'
        elif isinstance(action, MessageAction):
            return action.content
        elif isinstance(action, AgentFinishAction) and action.source == 'agent':
            return action.thought
        elif isinstance(action, RunRegressionAction):
            return f'{action.thought}\n<run_regression></run_regression>'
        return ''


class CodeActActionParserFinish(ActionParser):
    """Parser action:
    - AgentFinishAction() - end the interaction
    """

    def __init__(
        self,
    ):
        self.finish_command = None

    def check_condition(self, action_str: str) -> bool:
        self.finish_command = re.search(r'<finish>.*</finish>', action_str, re.DOTALL)
        return self.finish_command is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.finish_command is not None
        ), 'self.finish_command should not be None when parse is called'
        thought = action_str.replace(self.finish_command.group(0), '').strip()
        return AgentFinishAction(outputs={'thought': thought}, thought=thought)


class CodeActActionParserCmdRun(ActionParser):
    """Parser action:
    - CmdRunAction(command) - bash command to run
    - AgentFinishAction() - end the interaction
    """

    def __init__(
        self,
    ):
        self.bash_command = None

    def check_condition(self, action_str: str) -> bool:
        self.bash_command = re.search(
            r'<execute_bash>(.*?)</execute_bash>', action_str, re.DOTALL
        )
        return self.bash_command is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.bash_command is not None
        ), 'self.bash_command should not be None when parse is called'
        thought = action_str.replace(self.bash_command.group(0), '').strip()
        # a command was found
        command_group = self.bash_command.group(1).strip()
        if command_group.strip() == 'exit':
            return AgentFinishAction(thought=thought)
        return CmdRunAction(command=command_group, thought=thought)


class CodeActActionParserIPythonRunCell(ActionParser):
    """Parser action:
    - IPythonRunCellAction(code) - IPython code to run
    """

    def __init__(
        self,
    ):
        self.python_code = None
        self.jupyter_kernel_init_code: str = 'from agentskills import *'

    def check_condition(self, action_str: str) -> bool:
        self.python_code = re.search(
            r'<execute_ipython>(.*?)</execute_ipython>', action_str, re.DOTALL
        )
        return self.python_code is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.python_code is not None
        ), 'self.python_code should not be None when parse is called'
        code_group = self.python_code.group(1).strip()
        thought = action_str.replace(self.python_code.group(0), '').strip()
        return IPythonRunCellAction(
            code=code_group,
            thought=thought,
            kernel_init_code=self.jupyter_kernel_init_code,
        )


class CodeActActionParserAgentDelegate(ActionParser):
    """Parser action:
    - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
    """

    def __init__(
        self,
    ):
        self.agent_delegate = None

    def check_condition(self, action_str: str) -> bool:
        self.agent_delegate = re.search(
            r'<execute_browse>(.*)</execute_browse>', action_str, re.DOTALL
        )
        return self.agent_delegate is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.agent_delegate is not None
        ), 'self.agent_delegate should not be None when parse is called'
        thought = action_str.replace(self.agent_delegate.group(0), '').strip()
        browse_actions = self.agent_delegate.group(1).strip()
        thought = (
            f'{thought}\nI should start with: {browse_actions}'
            if thought
            else f'I should start with: {browse_actions}'
        )

        return AgentDelegateAction(
            agent='BrowsingAgent', thought=thought, inputs={'task': browse_actions}
        )


class CodeActActionParserMessage(ActionParser):
    """Parser action:
    - MessageAction(content) - Message action to run (e.g. ask for clarification)
    """

    def __init__(
        self,
    ):
        pass

    def check_condition(self, action_str: str) -> bool:
        # We assume the LLM is GOOD enough that when it returns pure natural language
        # it wants to talk to the user
        return True

    def parse(self, action_str: str) -> Action:
        return MessageAction(content=action_str, wait_for_response=True)


class CodeActActionParserFileEdit(ActionParser):
    """Parser action:
    - FileEditAction(path, content) - edit a file
    """

    def __init__(self):
        self.file_edit_match: re.Match | None = None

    def check_condition(self, action_str: str) -> bool:
        if '<file_edit' not in action_str:
            return False

        # Updated regex to make start and end optional
        self.file_edit_match = re.search(
            r'<file_edit\s+path=(["\']?)(.*?)\1(?:\s+start=(["\']?)(.*?)\3)?(?:\s+end=(["\']?)(.*?)\5)?\s*>(.*?)</file_edit>',
            # r'<file_edit\s+path=(["\']?)(.*?)\1\s+start=(["\']?)(.*?)\3\s+end=(["\']?)(.*?)\5\s*>(.*?)</file_edit>',
            action_str,
            re.DOTALL,
        )

        if self.file_edit_match is None:
            # logger.error(
            #     f'FileEditAction detected but the format is incorrect. Unable to match for <file_edit> in:\n{"-" * 80}\n{action_str}\n{"-" * 80}'
            # )
            # import pdb; pdb.set_trace()
            raise LLMMalformedActionError(
                message='FileEditAction detected but the format is incorrect. Usage:\n'
                '<file_edit path="[path]" start=[start_line or -1] end=[end_line or -1]>\n'
                '[content_to_edit]\n'
                '</file_edit>\n',
                action_str=action_str,
            )

        path = self.file_edit_match.group(2)
        start = self.file_edit_match.group(4)
        end = self.file_edit_match.group(6)

        if not path:
            raise LLMMalformedActionError(
                'FileEditAction detected but no `path` specified. You should specify the path of the file to edit.'
            )

        if start:
            try:
                int(start)
            except ValueError:
                raise LLMMalformedActionError(
                    f'FileEditAction detected but `start` is not a valid integer: {start}'
                )

        if end:
            try:
                int(end)
            except ValueError:
                raise LLMMalformedActionError(
                    f'FileEditAction detected but `end` is not a valid integer: {end}'
                )

        return True

    def parse(self, action_str: str) -> Action:
        assert (
            self.file_edit_match is not None
        ), 'self.file_edit_match should not be None when parse is called'

        file_path = self.file_edit_match.group(2).strip()
        start_line = (
            int(self.file_edit_match.group(4))
            if self.file_edit_match.group(4)
            else None
        )
        end_line = (
            int(self.file_edit_match.group(6))
            if self.file_edit_match.group(6)
            else None
        )
        content = self.file_edit_match.group(7)
        thought = action_str.replace(self.file_edit_match.group(0), '').strip()

        action = FileEditAction(path=file_path, content=content, thought=thought)
        if start_line is not None:
            action.start = start_line
        if end_line is not None:
            action.end = end_line
        return action


class CodeActActionParserRunRegression(ActionParser):
    """Parser action:
    - RunRegressionAction(repo, version, test_command, testcases) - run regression tests
    """

    def __init__(self, instance: dict[str, Any] | None):
        self.run_regression_command = None
        self.instance = instance

    def check_condition(self, action_str: str) -> bool:
        self.run_regression_match = re.search(
            r'<run_regression>.*</run_regression>', action_str, re.DOTALL
        )

        return self.run_regression_match is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.run_regression_match is not None
        ), 'self.run_regression_match should not be None when parse is called'

        thought = action_str.replace(self.run_regression_match.group(0), '').strip()

        instance = self.instance
        assert instance, 'Instance metadata must be set if regression is enabled.'
        instance_id = instance['instance_id']

        # Convert e.g. "logs/scikit-learn__scikit-learn-12421/test_output.txt" to "scikit-learn/scikit-learn"
        repo = '-'.join(
            instance_id.replace('__', '/').split('-')[:-1]
        )  # e.g. scikit-learn/scikit-learn

        test_command = ' '.join(
            [
                MAP_REPO_VERSION_TO_SPECS[instance['repo']][instance['version']][
                    'test_cmd'
                ],
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
            thought=thought,
        )
        return action
