import json
import os
from collections import deque
from itertools import islice
from typing import Any

from litellm import ModelResponse

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.agenthub.codeact_agent.action_parser import CodeActResponseParser
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import ImageContent, Message, TextContent
from openhands.events.action import (
    Action,
    AgentDelegateAction,
    AgentFinishAction,
    CmdRunAction,
    FileEditAction,
    IPythonRunCellAction,
    MessageAction,
)
from openhands.events.observation import (
    AgentDelegateObservation,
    CmdOutputObservation,
    FileEditObservation,
    IPythonRunCellObservation,
    UserRejectObservation,
)
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.observation import Observation
from openhands.events.serialization.event import truncate_content
from openhands.llm.llm import LLM
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.microagent import MicroAgent
from openhands.utils.prompt import PromptManager


class CodeActAgentEdit(Agent):
    VERSION = '2.1'
    """
    The Code Act Agent is a minimalist agent.
    The agent works by passing the model a list of action-observation pairs and prompting the model to take the next step.

    ### Overview

    This agent implements the CodeAct idea ([paper](https://arxiv.org/abs/2402.01030), [tweet](https://twitter.com/xingyaow_/status/1754556835703751087)) that consolidates LLM agents’ **act**ions into a unified **code** action space for both *simplicity* and *performance* (see paper for more details).

    The conceptual idea is illustrated below. At each turn, the agent can:

    1. **Converse**: Communicate with humans in natural language to ask for clarification, confirmation, etc.
    2. **CodeAct**: Choose to perform the task by executing code
    - Execute any valid Linux `bash` command
    - Execute any valid `Python` code with [an interactive Python interpreter](https://ipython.org/). This is simulated through `bash` command, see plugin system below for more details.

    ![image](https://github.com/All-Hands-AI/OpenHands/assets/38853559/92b622e3-72ad-4a61-8f41-8c040b6d5fb3)

    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]
    obs_prefix = 'OBSERVATION:\n'

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
    ) -> None:
        """Initializes a new instance of the CodeActAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm, config)
        self.reset()

        self.micro_agent = (
            MicroAgent(
                os.path.join(
                    os.path.dirname(__file__), 'micro', f'{config.micro_agent_name}.md'
                )
            )
            if config.micro_agent_name
            else None
        )
        self.config.function_calling = False
        if (
            self.config.function_calling
            and not self.llm.config.supports_function_calling
        ):
            logger.warning(
                f'Function calling not supported for model {self.llm.config.model}. '
                'Disabling function calling.'
            )
            self.config.function_calling = False

        if self.config.function_calling:
            # Function calling mode
            self.tools = codeact_function_calling.get_tools(
                codeact_enable_browsing_delegate=self.config.codeact_enable_browsing_delegate,
                codeact_enable_jupyter=self.config.codeact_enable_jupyter,
                codeact_enable_llm_editor=self.config.codeact_enable_llm_editor,
            )
            logger.info(
                f'TOOLS loaded for CodeActAgent: {json.dumps(self.tools, indent=2)}'
            )
            self.system_prompt = codeact_function_calling.SYSTEM_PROMPT
            self.initial_user_message = None
        else:
            # Non-function-calling mode
            # import pdb; pdb.set_trace()
            self.action_parser = CodeActResponseParser()
            self.prompt_manager = PromptManager(
                prompt_dir=os.path.join(os.path.dirname(__file__)),
                agent_skills_docs=AgentSkillsRequirement.documentation,
                micro_agent=self.micro_agent,
            )
            self.system_prompt = self.prompt_manager.system_message
            self.initial_user_message = self.prompt_manager.initial_user_message
        logger.info(f'System prompt: {self.system_prompt}')
        logger.info(f'Initial user message: {self.initial_user_message}')
        self.best_of_n = 1
        if self.best_of_n > 1:
            from openai import OpenAI

            api_key = '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'
            base_url = 'https://api.openai.com/v1'
            self.embedding_client = OpenAI(base_url=base_url, api_key=api_key)
            self.embedding_model = 'text-embedding-3-large'

        self.pending_actions: deque[Action] = deque()
        self.extra_data: dict[str, Any] = {}

    def get_action_message(
        self,
        action: Action,
        pending_tool_call_action_messages: dict[str, Message],
    ) -> list[Message]:
        """Converts an action into a message format that can be sent to the LLM.

        This method handles different types of actions and formats them appropriately:
        1. For tool-based actions (AgentDelegate, CmdRun, IPythonRunCell, FileEdit) and agent-sourced AgentFinish:
            - In function calling mode: Stores the LLM's response in pending_tool_call_action_messages
            - In non-function calling mode: Creates a message with the action string
        2. For MessageActions: Creates a message with the text content and optional image content

        Args:
            action (Action): The action to convert. Can be one of:
                - AgentDelegateAction: For delegating tasks to other agents
                - CmdRunAction: For executing bash commands
                - IPythonRunCellAction: For running IPython code
                - FileEditAction: For editing files
                - AgentFinishAction: For ending the interaction
                - MessageAction: For sending messages
            pending_tool_call_action_messages (dict[str, Message]): Dictionary mapping response IDs
                to their corresponding messages. Used in function calling mode to track tool calls
                that are waiting for their results.

        Returns:
            list[Message]: A list containing the formatted message(s) for the action.
                May be empty if the action is handled as a tool call in function calling mode.

        Note:
            In function calling mode, tool-based actions are stored in pending_tool_call_action_messages
            rather than being returned immediately. They will be processed later when all corresponding
            tool call results are available.
        """
        # create a regular message from an event
        if isinstance(
            action,
            (
                AgentDelegateAction,
                CmdRunAction,
                IPythonRunCellAction,
                FileEditAction,
            ),
        ) or (isinstance(action, AgentFinishAction) and action.source == 'agent'):
            if self.config.function_calling:
                tool_metadata = action.tool_call_metadata
                assert tool_metadata is not None, (
                    'Tool call metadata should NOT be None when function calling is enabled. Action: '
                    + str(action)
                )

                llm_response: ModelResponse = tool_metadata.model_response
                assistant_msg = llm_response.choices[0].message
                # Add the LLM message (assistant) that initiated the tool calls
                # (overwrites any previous message with the same response_id)
                pending_tool_call_action_messages[llm_response.id] = Message(
                    role=assistant_msg.role,
                    # tool call content SHOULD BE a string
                    content=[TextContent(text=assistant_msg.content)]
                    if assistant_msg.content is not None
                    else [],
                    tool_calls=assistant_msg.tool_calls,
                )
                return []
            else:
                content = [TextContent(text=self.action_parser.action_to_str(action))]
                return [
                    Message(
                        role='user' if action.source == 'user' else 'assistant',
                        content=content,
                    )
                ]
        elif isinstance(action, MessageAction):
            role = 'user' if action.source == 'user' else 'assistant'
            content = [TextContent(text=action.content)]
            if self.llm.vision_is_active() and action.images_urls:
                content.append(ImageContent(image_urls=action.images_urls))
            return [
                Message(
                    role=role,
                    content=content,
                )
            ]
        return []

    def get_observation_message(
        self,
        obs: Observation,
        tool_call_id_to_message: dict[str, Message],
    ) -> list[Message]:
        """Converts an observation into a message format that can be sent to the LLM.

        This method handles different types of observations and formats them appropriately:
        - CmdOutputObservation: Formats command execution results with exit codes
        - IPythonRunCellObservation: Formats IPython cell execution results, replacing base64 images
        - FileEditObservation: Formats file editing results
        - AgentDelegateObservation: Formats results from delegated agent tasks
        - ErrorObservation: Formats error messages from failed actions
        - UserRejectObservation: Formats user rejection messages

        In function calling mode, observations with tool_call_metadata are stored in
        tool_call_id_to_message for later processing instead of being returned immediately.

        Args:
            obs (Observation): The observation to convert
            tool_call_id_to_message (dict[str, Message]): Dictionary mapping tool call IDs
                to their corresponding messages (used in function calling mode)

        Returns:
            list[Message]: A list containing the formatted message(s) for the observation.
                May be empty if the observation is handled as a tool response in function calling mode.

        Raises:
            ValueError: If the observation type is unknown
        """
        message: Message
        max_message_chars = self.llm.config.max_message_chars
        obs_prefix = 'OBSERVATION:\n'
        if isinstance(obs, CmdOutputObservation):
            text = obs_prefix + truncate_content(
                obs.content + obs.interpreter_details, max_message_chars
            )
            text += f'\n[Command finished with exit code {obs.exit_code}]'
            message = Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, IPythonRunCellObservation):
            text = obs_prefix + obs.content
            # replace base64 images with a placeholder
            splitted = text.split('\n')
            for i, line in enumerate(splitted):
                if '![image](data:image/png;base64,' in line:
                    splitted[i] = (
                        '![image](data:image/png;base64, ...) already displayed to user'
                    )
            text = '\n'.join(splitted)
            text = truncate_content(text, max_message_chars)
            message = Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, FileEditObservation):
            text = obs_prefix + truncate_content(str(obs), max_message_chars)
            message = Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, AgentDelegateObservation):
            text = obs_prefix + truncate_content(
                obs.outputs['content'] if 'content' in obs.outputs else '',
                max_message_chars,
            )
            message = Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, ErrorObservation):
            text = obs_prefix + truncate_content(obs.content, max_message_chars)
            text += '\n[Error occurred in processing last action]'
            message = Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, UserRejectObservation):
            text = 'OBSERVATION:\n' + truncate_content(obs.content, max_message_chars)
            text += '\n[Last action has been rejected by the user]'
            message = Message(role='user', content=[TextContent(text=text)])
        else:
            # If an observation message is not returned, it will cause an error
            # when the LLM tries to return the next message
            raise ValueError(f'Unknown observation type: {type(obs)}')

        if self.config.function_calling:
            # Update the message as tool response properly
            if (tool_call_metadata := obs.tool_call_metadata) is not None:
                tool_call_id_to_message[tool_call_metadata.tool_call_id] = Message(
                    role='tool',
                    content=message.content,
                    tool_call_id=tool_call_metadata.tool_call_id,
                    name=tool_call_metadata.function_name,
                )
                # No need to return the observation message
                # because it will be added by get_action_message when all the corresponding
                # tool calls in the SAME request are processed
                return []

        return [message]

    def reset(self) -> None:
        """Resets the CodeAct Agent."""
        super().reset()

    def step(self, state: State) -> Action:
        """Performs one step using the CodeAct Agent.
        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - CmdRunAction(command) - bash command to run
        - IPythonRunCellAction(code) - IPython code to run
        - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        """

        def check_tool_calls(choice, tools):
            if choice.message.tool_calls is None:
                return True
            if tools is None:
                return True
            if all(
                [
                    tool_call.function.name in tools
                    for tool_call in choice.message.tool_calls
                ]
            ):
                return True
            return False

        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.history.get_last_user_message()
        if latest_user_message and latest_user_message.strip() == '/exit':
            return AgentFinishAction()

        # prepare what we want to send to the LLM
        messages = self._get_messages(state)
        messages = self._post_process_messages(messages)
        params: dict = {
            'messages': self.llm.format_messages_for_llm(messages),
            'n': self.best_of_n,
        }
        if self.config.function_calling:
            params['tools'] = self.tools
        else:
            pass
            # params['stop'] = [
            #     '</execute_ipython>',
            #     '</execute_bash>',
            #     '</execute_browse>',
            #     '</file_edit>',
            # ]

        if (
            len(messages) > 4
            and messages[-1].role == 'user'
            and "You've been working on this task for a while."
            in messages[-1].content[0].text
        ):
            params['tools'] = None

        # response = self.llm.completion(**params)
        # response = self.choices_selection(response)

        # import pdb; pdb.set_trace()
        correct_response = None
        tools_names = (
            [tool['function']['name'] for tool in self.tools]
            if self.config.function_calling
            else None
        )
        for i in range(5):
            # import pdb; pdb.set_trace()
            response = self.llm.completion(**params)
            self.extra_data['input_tokens'] = response.usage.get('prompt_tokens')
            self.extra_data['output_tokens'] = response.usage.get('completion_tokens')
            response.choices = [
                choice
                for choice in response.choices
                if choice.message.content is not None
                or choice.message.tool_calls is not None
            ]
            if len(response.choices) == 0:
                continue

            for choice in response.choices:
                if check_tool_calls(choice, tools_names):
                    correct_response = response.__deepcopy__()
                    correct_response.choices = [choice]

            response = self.choices_selection_llm(
                messages[-1].content[0].text, response
            )
            # response = self.choices_selection_embedding(response)

            if check_tool_calls(response.choices[0], tools_names):
                break
            logger.info(f'Tool calls not correct in the response, retrying {i} ...')

        if not check_tool_calls(response.choices[0], tools_names) and correct_response:
            logger.info(
                'Tool calls not correct in the response, replace with the correct one.'
            )
            response = correct_response

        if self.config.function_calling:
            actions = codeact_function_calling.response_to_actions(response)
            for action in actions:
                self.pending_actions.append(action)
            return self.pending_actions.popleft()
        else:
            return self.action_parser.parse(response)

    def finish(self, state: State, finish_reason: str = None) -> Action:
        # prepare what we want to send to the LLM
        messages = self._get_messages(state)
        messages = self._post_process_messages(messages)
        if finish_reason:
            finish_prompt = (
                f"You are required to stop your work due to: \n<stop_reason>\n{finish_reason}\n</stop_reason>\n\n"
                "Before exiting, please provide a detailed and structured summary of your work. Your summary should include the following components:\n\n"
                "1. **Reason for Stopping**: Briefly explain the reason for stopping the task, as specified by <stop_reason>, and how it impacted your progress or completion of the task. \n\n"
                "2. **Overview of Work Completed**:\n"
                "   - Summarize the objectives of your work and the progress made so far.\n"
                "   - Clearly state what has been fully completed, partially completed, or left unaddressed.\n\n"
                "3. **Code Changes**:\n"
                "   - List all the files you have worked on.\n"
                "   - Specify the line numbers or sections where changes were made.\n"
                "   - Provide a brief explanation of each change, including its purpose and functionality.\n"
                "   - Provide the exact code you have edited with the file name and line number in the format of ```\n<code>\n```, so that the next engineer can directly know what you have done.\n\n"
                "4. **Relevant Code**:\n"
                "   - Identify any additional code, files, or modules (even if unedited) that are relevant to your work.\n"
                "   - Include file names and line numbers, and explain their significance in the context of your task.\n"
                "   - Provide the exact relevant code with the file name and line number in the format of ```\n<code>\n```, so that the next engineer can directly see the code.\n\n"
                "5. **Challenges or Issues**:\n"
                "   - Highlight any challenges, blockers, or unresolved issues encountered during your work.\n"
                "   - If applicable, provide suggestions or potential solutions for these issues.\n\n"
                "6. **Next Steps for Engineers**:\n"
                "   - Provide recommendations for the next engineer who will continue this work.\n"
                "   - Include guidance on how to pick up where you left off, any warnings or caveats, and potential areas for improvement or optimization.\n\n"
                "Be as thorough and detailed as possible to ensure a smooth transition and clear understanding of your work."
            )
        else:
            finish_prompt = (
                "Your task is complete. Please provide a detailed summary of your work before exiting. Your summary should include the following:\n\n"
                "1. **Overview of the Work**: Provide a clear and concise explanation of what you were tasked with, the goals you aimed to achieve, and the context of your work.\n\n"
                "2. **Code Changes**:\n"
                "   - List all the files you have worked on.\n"
                "   - Specify the line numbers or sections where changes were made.\n"
                "   - Provide a brief explanation of each change, including its purpose and functionality.\n"
                "   - Provide the exact code you have edited with the file name and line number in the format of ```\n<code>\n```, so that the next engineer can directly know what you have done.\n\n"
                "3. **Relevant Code**:\n"
                "   - Identify any additional code, files, or modules (even if unedited) that are relevant to your work.\n"
                "   - Include file names and line numbers, and explain their significance in the context of your task.\n"
                "   - Provide the exact relevant code with the file name and line number in the format of ```\n<code>\n```, so that the next engineer can directly see the code.\n\n"
                "4 **Next Steps for Engineers**: Provide guidance for the next engineer who will work on this project, including:\n"
                "   - Any unresolved issues or potential areas for improvement.\n"
                "   - Suggestions for future development or maintenance.\n"
                "   - Any specific instructions or warnings about the code or system.\n\n"
                "Be as thorough and detailed as possible to ensure a smooth handoff to the next engineer."
            )
            
        if messages[-1].role == 'user':
            messages[-1].content[0].text += '\n\n' + finish_prompt
        else:
            messages.append(Message(role='user', content=[TextContent(text=finish_prompt)]))
            
        params: dict = {
            'messages': self.llm.format_messages_for_llm(messages),
        }
    
        response = self.llm.completion(**params)
        self.extra_data['input_tokens'] = response.usage.get('prompt_tokens')
        self.extra_data['output_tokens'] = response.usage.get('completion_tokens')
        
        thought = response.choices[0].message.content.replace(finish_prompt, '').strip()
        return AgentFinishAction(outputs={'thought': thought}, thought=thought)

    def choices_selection_llm(
        self, question: str, response: ModelResponse
    ) -> ModelResponse:
        # import pdb; pdb.set_trace()
        if len(response.choices) == 1:
            return response
        assert (
            not self.config.function_calling
        ), 'choices_selection_llm is only supported in non-function calling mode'
        options_list = [
            f"Answer Option {i}:\n```\n{choice.message.to_dict()['content']}\n```"
            for i, choice in enumerate(response.choices)
        ]
        index_dict = {i: 0 for i in range(len(options_list))}
        options_list = '\n'.join(options_list)  # type: ignore
        prompt = (
            'Please evaluate the given question and select the best option from the provided choices.\n'
            'The best option is the one that most effectively solves the problem or best addresses the question. \n'
            'Instructions:\n'
            '1. Carefully analyze the content and differences between the provided options.\n'
            '2. Select the most appropriate option based on the information given.\n'
            '3. Provide a clear explanation for your selection, highlighting why it is the best choice.\n'
            '4. At the end of your response, return the index of the selected option in the following format: `[[index]]` (e.g., `[[0]]` for the first option).\n'
            'Here is the question:\n'
            '```\n'
            f'{question}\n'
            '```\n'
            'Here are the options:\n'
            f'{options_list}\n'
            'Your response should strictly follow the required format: `[[index]]`.'
        )
        params: dict = {
            'messages': self.llm.format_messages_for_llm(
                [
                    Message(
                        role='user',
                        content=[TextContent(text=prompt)],
                    )
                ]
            ),
            'n': 5,
        }
        options_response = self.llm.completion(**params)
        option_texts = [choice.message.content for choice in options_response.choices]
        for option_text in option_texts:
            try:
                index = int(option_text.split('[[')[1].split(']]')[0])
                index_dict[index] += 1
            except Exception as e:
                logger.info(f'Error selecting the best option: {e}')
        logger.info(f'Select the best choice: {index_dict}')
        index = max(index_dict, key=index_dict.get)  # type: ignore
        response.choices = [response.choices[index]]
        return response

    def choices_selection_embedding(self, response: ModelResponse) -> ModelResponse:
        if len(response.choices) == 1:
            return response

        # response.choices = [response.choices[0]]
        choices = [choice.message.to_dict() for choice in response.choices]
        for choice in choices:
            if isinstance(choice['tool_calls'], list):
                [call.pop('id') for call in choice['tool_calls']]
        choices = [
            json.dumps(choice, indent=4, ensure_ascii=False) for choice in choices
        ]
        # choices = [choice[:24000] for choice in choices]

        try:
            embedding_response = self.embedding_client.embeddings.create(
                input=choices,
                model=self.embedding_model,
                timeout=120,
            )

            embeddings = [embedding.embedding for embedding in embedding_response.data]
            # import pdb; pdb.set_trace()
            closest_index, labels = codeact_function_calling.select_by_kmeans(
                embeddings
            )
            logger.info(f'Select the best choice: {closest_index}, labels: {labels}')
            # closest_index = codeact_function_calling.select_by_dbscan(embeddings)
            # closest_index = codeact_function_calling.select_by_optics(embeddings)
        except Exception as e:
            logger.error(f'Error embedding choices: {e}')
            closest_index = 0

        # 更新response.choices为离类中心最近的选项
        response.choices = [response.choices[closest_index]]

        return response

    def _post_process_messages(self, messages: list[Message]) -> list[Message]:
        # import pdb; pdb.set_trace()
        first_opservation_index = -1
        opservation_role = 'tool' if self.config.function_calling else 'assistant'
        for i, message in enumerate(messages):
            if message.role == opservation_role:
                first_opservation_index = i
                break
        if first_opservation_index == -1:
            return messages

        summary_user_message_index = []
        for i in range(len(messages)):
            if (
                messages[i].role == 'user'
                and "You've been working on this task for a while."
                in messages[i].content[0].text
            ):
                summary_user_message_index.append(i)
        # import pdb; pdb.set_trace()
        if len(summary_user_message_index) <= 1:
            return messages

        # if summary_user_message_index[-1] == len(messages) - 1 and len(summary_user_message_index) == 1:
        #     return messages
        # elif summary_user_message_index[-1] == len(messages) - 1:
        if summary_user_message_index[-1] == len(messages) - 1:
            # import pdb; pdb.set_trace()
            return (
                messages[: first_opservation_index + 1]
                + messages[summary_user_message_index[-2] :]
            )
        else:
            # import pdb; pdb.set_trace()
            return (
                messages[: first_opservation_index + 1]
                + messages[summary_user_message_index[-1] :]
            )

    def _post_process_messages_desc(
        self, messages: list[Message], max_message_iterations: int = -1
    ) -> list[Message]:
        if max_message_iterations < 0:
            return messages
        else:
            post_processed_messages = []
            if len(messages) > 0:
                if messages[0].role == 'system':
                    post_processed_messages.append(messages[0])
                else:
                    raise ValueError(
                        f'The first message must be a system message, but it is {messages[0].role}'
                    )
            if len(messages) > 1:
                if messages[1].role == 'user':
                    post_processed_messages.append(messages[1])
                else:
                    raise ValueError(
                        f'The second message must be a user message, but it is {messages[1].role}'
                    )
            if len(messages) > 2:
                if messages[2].role == 'assistant':
                    post_processed_messages.append(messages[2])
                else:
                    raise ValueError(
                        f'The third message must be an assistant message, but it is {messages[2].role}'
                    )
            if len(messages) > 3 and messages[3].role == 'tool':
                post_processed_messages.append(messages[3])

            if len(messages) > len(post_processed_messages) + max_message_iterations:
                post_processed_messages.append(
                    Message(
                        role='user',
                        content=[
                            TextContent(text="You are doing great! Let's continue.")
                        ],
                    )
                )
                while messages[-max_message_iterations].role != 'assistant':
                    max_message_iterations -= 1
                post_processed_messages.extend(messages[-max_message_iterations:])
            else:
                post_processed_messages.extend(messages[len(post_processed_messages) :])

            return post_processed_messages

    def _get_messages(self, state: State) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand. It handles both regular
        message flow and function-calling scenarios.

        The method performs the following steps:
        1. Initializes with system prompt and optional initial user message
        2. Processes events (Actions and Observations) into messages
        3. Handles tool calls and their responses in function-calling mode
        4. Manages message role alternation (user/assistant/tool)
        5. Applies caching for specific LLM providers (e.g., Anthropic)
        6. Adds environment reminders for non-function-calling mode

        Args:
            state (State): The current state object containing conversation history and other metadata

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption, including:
                - System message with prompt
                - Initial user message (if configured)
                - Action messages (from both user and assistant)
                - Observation messages (including tool responses)
                - Environment reminders (in non-function-calling mode)

        Note:
            - In function-calling mode, tool calls and their responses are carefully tracked
              to maintain proper conversation flow
            - Messages from the same role are combined to prevent consecutive same-role messages
            - For Anthropic models, specific messages are cached according to their documentation
        """
        messages: list[Message] = [
            Message(
                role='system',
                content=[
                    TextContent(
                        text=self.system_prompt,
                        cache_prompt=self.llm.is_caching_prompt_active(),  # Cache system prompt
                    )
                ],
            )
        ]
        if self.initial_user_message:
            messages.append(
                Message(
                    role='user',
                    content=[TextContent(text=self.initial_user_message)],
                )
            )

        pending_tool_call_action_messages: dict[str, Message] = {}
        tool_call_id_to_message: dict[str, Message] = {}
        events = list(state.history.get_events())
        for event in events:
            # create a regular message from an event
            if isinstance(event, Action):
                messages_to_add = self.get_action_message(
                    action=event,
                    pending_tool_call_action_messages=pending_tool_call_action_messages,
                )
            elif isinstance(event, Observation):
                messages_to_add = self.get_observation_message(
                    obs=event,
                    tool_call_id_to_message=tool_call_id_to_message,
                )
            else:
                raise ValueError(f'Unknown event type: {type(event)}')

            # Check pending tool call action messages and see if they are complete
            _response_ids_to_remove = []
            for (
                response_id,
                pending_message,
            ) in pending_tool_call_action_messages.items():
                assert pending_message.tool_calls is not None, (
                    'Tool calls should NOT be None when function calling is enabled & the message is considered pending tool call. '
                    f'Pending message: {pending_message}'
                )
                if all(
                    tool_call.id in tool_call_id_to_message
                    for tool_call in pending_message.tool_calls
                ):
                    # If complete:
                    # -- 1. Add the message that **initiated** the tool calls
                    messages_to_add.append(pending_message)
                    # -- 2. Add the tool calls **results***
                    for tool_call in pending_message.tool_calls:
                        messages_to_add.append(tool_call_id_to_message[tool_call.id])
                        tool_call_id_to_message.pop(tool_call.id)
                    _response_ids_to_remove.append(response_id)
            # Cleanup the processed pending tool messages
            for response_id in _response_ids_to_remove:
                pending_tool_call_action_messages.pop(response_id)

            for message in messages_to_add:
                # add regular message
                if message:
                    # handle error if the message is the SAME role as the previous message
                    # litellm.exceptions.BadRequestError: litellm.BadRequestError: OpenAIException - Error code: 400 - {'detail': 'Only supports u/a/u/a/u...'}
                    # there shouldn't be two consecutive messages from the same role
                    # NOTE: we shouldn't combine tool messages because each of them has a different tool_call_id
                    if (
                        messages
                        and messages[-1].role == message.role
                        and message.role != 'tool'
                    ):
                        # messages[-1].content.extend(message.content)
                        messages[-1].content = message.content
                    else:
                        messages.append(message)

        if self.llm.is_caching_prompt_active():
            # NOTE: this is only needed for anthropic
            # following logic here:
            # https://github.com/anthropics/anthropic-quickstarts/blob/8f734fd08c425c6ec91ddd613af04ff87d70c5a0/computer-use-demo/computer_use_demo/loop.py#L241-L262
            breakpoints_remaining = 3  # remaining 1 for system/tool
            for message in reversed(messages):
                if message.role == 'user' or message.role == 'tool':
                    if breakpoints_remaining > 0:
                        message.content[
                            -1
                        ].cache_prompt = True  # Last item inside the message content
                        breakpoints_remaining -= 1
                    else:
                        break

        if not self.config.function_calling:
            # The latest user message is important:
            # we want to remind the agent of the environment constraints
            latest_user_message = next(
                islice(
                    (
                        m
                        for m in reversed(messages)
                        if m.role == 'user'
                        and any(isinstance(c, TextContent) for c in m.content)
                    ),
                    1,
                ),
                None,
            )
            # do not add this for function calling
            if latest_user_message:
                reminder_text = f'\n\nENVIRONMENT REMINDER: You have {state.max_iterations - state.iteration} turns left to complete the task. When finished reply with <finish></finish>.'
                latest_user_message.content[0].text += reminder_text

        return messages
