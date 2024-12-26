from dataclasses import dataclass

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class RunRegressionObservation(Observation):
    """This data class represents the output of a command."""

    passed: bool
    report: dict[str, str]
    output: str
    error_output: str
    tests_not_passed: list[str]
    testcase_output: dict[str, str]

    test_command: str

    exit_code: int = 0
    observation: str = ObservationType.RUN_REGRESSION

    @property
    def error(self) -> bool:
        return self.exit_code != 0

    @property
    def message(self) -> str:
        return (
            f'Command `{self.test_command}` executed with exit code {self.exit_code}.'
        )

    def __str__(self) -> str:
        return f'**RunRegressionObservation (source={self.source}, exit code={self.exit_code})**\nContent:\n{self.content}\n'
