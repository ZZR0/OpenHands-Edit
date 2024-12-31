from dataclasses import dataclass

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class UnknownActionObservation(Observation):
    """This data class represents an error encountered by the agent.

    This is the type of error that LLM can recover from.
    E.g., Linter error after editing a file.
    """

    observation: str = ObservationType.UNKNOWN

    @property
    def message(self) -> str:
        return self.content

    def __str__(self) -> str:
        return f'**UnknownActionObservation**\n{self.content}'
