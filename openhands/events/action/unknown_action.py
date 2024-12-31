from dataclasses import dataclass

from openhands.core.schema import ActionType
from openhands.events.action.action import (
    Action,
)


@dataclass
class UnknownAction(Action):
    tool_name: str
    thought: str = ''
    action: str = ActionType.UNKNOWN
    tool_list: list[str] | None = None

    @property
    def message(self) -> str:
        res = f'Unknown tool call: `{self.tool_name}`\nPlease check if the tool name is in the tool list.\n'
        res += 'Maybe this tool is a secondary tool? If so, please use the primary tool name, and follow the format described in the tool list.\n'
        if self.tool_list:
            res += 'Allowed Tool list: '
            for i, tool in enumerate(self.tool_list):
                if i:
                    res += ', '
                res += f'`{tool}`'
            res += '\n'
        return res

    def __str__(self) -> str:
        res = f'**UnknownAction**\nTOOL: {self.tool_name}\n'
        if self.thought:
            res += f'THOUGHT: {self.thought}\n'
        return res
