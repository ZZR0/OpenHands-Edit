from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.agenthub.codeact_agent.codeact_agent_edit import CodeActAgentEdit
from openhands.controller.agent import Agent

Agent.register('CodeActAgent', CodeActAgent)
Agent.register('CodeActAgentEdit', CodeActAgentEdit)
