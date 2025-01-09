from dataclasses import dataclass
from typing import ClassVar

from openhands.core.schema import ActionType
from openhands.events.action.action import (
    Action,
    ActionConfirmationStatus,
    ActionSecurityRisk,
)


@dataclass
class RunRegressionAction(Action):
    repo: str
    version: str
    test_command: str

    thought: str = ''
    _timeout: int = 300
    testcases: list[str] | None = None

    hidden: bool = False
    action: str = ActionType.RUN_REGRESSION
    runnable: ClassVar[bool] = True
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return f'Running regression tests with command: `{self.test_command}` for repo: `{self.repo}` version `{self.version}`'

    def __str__(self) -> str:
        ret = f'**RunRegressionAction (source={self.source})**\n'
        ret += f'REPO: {self.repo}\n'
        ret += f'VERSION: {self.version}\n'
        ret += f'COMMAND: {self.test_command}\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        return ret


# @dataclass
# class IPythonRunCellAction(Action):
#     code: str
#     thought: str = ''
#     include_extra: bool = (
#         True  # whether to include CWD & Python interpreter in the output
#     )
#     action: str = ActionType.RUN_IPYTHON
#     runnable: ClassVar[bool] = True
#     confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
#     security_risk: ActionSecurityRisk | None = None
#     kernel_init_code: str = ''  # code to run in the kernel (if the kernel is restarted)

#     def __str__(self) -> str:
#         ret = '**IPythonRunCellAction**\n'
#         if self.thought:
#             ret += f'THOUGHT: {self.thought}\n'
#         ret += f'CODE:\n{self.code}'
#         return ret

#     @property
#     def message(self) -> str:
#         return f'Running Python code interactively: {self.code}'
