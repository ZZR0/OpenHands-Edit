import copy
import os
import re
import tempfile
from abc import ABC, abstractmethod

from openhands.core.config import AppConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    FileEditAction,
    FileReadAction,
    FileWriteAction,
)
from openhands.events.observation import (
    ErrorObservation,
    FatalErrorObservation,
    FileEditObservation,
    FileReadObservation,
    FileWriteObservation,
    Observation,
)
from openhands.linter import DefaultLinter
from openhands.llm.llm import LLM
from openhands.llm.metrics import Metrics
from openhands.utils.chunk_localizer import Chunk, get_top_k_chunk_matches
from openhands.utils.diff import get_diff

SYS_MSG = """Your job is to produce a new version of the file based on the old version and the
provided draft of the new version. The provided draft may be incomplete (it may skip lines) and/or incorrectly indented. You should try to apply the changes present in the draft to the old version, and output a new version of the file.
NOTE:
- The output file should be COMPLETE and CORRECTLY INDENTED. Do not omit any lines, and do not change any lines that are not part of the changes.
- You should output the new version of the file by wrapping the new version of the file content in a ``` block.
- If there's no explicit comment to remove the existing code, we should keep them and append the new code to the end of the file.
- If there's placeholder comments like `# no changes before` or `# no changes here`, we should replace these comments with the original code near the placeholder comments.
"""

USER_MSG = """
HERE IS THE OLD VERSION OF THE FILE:
```
{old_contents}
```
HERE IS THE THOUGHT OF THE AGENT:
```
{thought}
```

HERE IS THE DRAFT OF THE NEW VERSION OF THE FILE:
```
{draft_changes}
```

GIVE ME THE NEW VERSION OF THE FILE.
IMPORTANT:
- There should be NO placeholder comments like `# no changes before` or `# no changes here`. They should be replaced with the original code near the placeholder comments.
- The output file should be COMPLETE and CORRECTLY INDENTED. Do not omit any lines, and do not change any lines that are not part of the changes.
""".strip()


def _extract_code(string):
    pattern = r'```(?:\w*\n)?(.*?)```'
    matches = re.findall(pattern, string, re.DOTALL)
    if not matches:
        return None
    return matches[0]


def get_new_file_contents(
    llm: LLM, old_contents: str, draft_changes: str, thought: str, num_retries: int = 3
) -> str | None:
    while num_retries > 0:
        messages = [
            {'role': 'system', 'content': SYS_MSG},
            {
                'role': 'user',
                'content': USER_MSG.format(
                    old_contents=old_contents, draft_changes=draft_changes, thought=thought
                ),
            },
        ]
        resp = llm.completion(messages=messages)
        new_contents = _extract_code(resp['choices'][0]['message']['content'])
        if new_contents is not None:
            return new_contents
        num_retries -= 1
    return None

def get_start_end_range(llm: LLM, thought: str) -> tuple[int, int]:
    # Define the system and user messages
    sys_msg = (
        "You are an assistant designed to extract the start and end line numbers of a code edit based on a provided thought. The thought contains an explanation or context about the code changes. "
        "Your task is to analyze the thought, identify the lines that are being edited, and return the start and end line numbers of the code edit. Please format the result as [START:<index>] for the start line number and [END:<index>] for the end line number, making it easy to parse programmatically. "
        "If no line numbers are found, return [START:-1] and [END:-1]."
    )
    
    user_msg = (
        "The draft contains one component: the thought. The thought provides context or explanation about the code changes. \n"   
        "Please extract the start and end line numbers from the thought, and return them in the format [START:index] and [END:index], where `index` is the corresponding line number. This will make it easier for me to parse. \n"
        f"Here is the agent's thought: \n{thought}"
    )

    messages = [
        {'role': 'system', 'content': sys_msg},
        {'role': 'user', 'content': user_msg},
    ]
    
    resp = llm.completion(messages=messages)
    start_pattern = r'\[START:(-?\d+)\]'
    end_pattern = r'\[END:(-?\d+)\]'
    start_match = re.search(start_pattern, resp['choices'][0]['message']['content'])
    end_match = re.search(end_pattern, resp['choices'][0]['message']['content'])
    if start_match and end_match:
        return int(start_match.group(1)), int(end_match.group(1))
    return None, None

def get_relevant_snippets(action: FileEditAction, original_file_content: str, max_lines_to_edit: int) -> str:
    # search for relevant ranges to hint the agent
    topk_chunks: list[Chunk] = get_top_k_chunk_matches(
        text=original_file_content,
        query=action.content,  # edit draft as query
        k=3,
        max_chunk_size=20,  # lines
    )
    error_msg = (
        'Here are some snippets that maybe relevant to the provided edit.\n'
    )
    for i, chunk in enumerate(topk_chunks):
        error_msg += f'[begin relevant snippet {i+1}. Line range: L{chunk.line_range[0]}-L{chunk.line_range[1]}. Similarity: {chunk.normalized_lcs}]\n'
        error_msg += f'[Browse around it via `open_file("{action.path}", {(chunk.line_range[0] + chunk.line_range[1]) // 2})`]\n'
        error_msg += chunk.visualize() + '\n'
        error_msg += f'[end relevant snippet {i+1}]\n'
        error_msg += '-' * 40 + '\n'

    error_msg += 'Consider using `open_file` to explore around the relevant snippets if needed.\n'
    error_msg += f'**IMPORTANT**: Please REDUCE the range of edits to less than {max_lines_to_edit} lines by setting `start` and `end` in the edit action (e.g. `<file_edit path="{action.path}" start=[PUT LINE NUMBER HERE] end=[PUT LINE NUMBER HERE] />`). '
    return error_msg


class FileEditRuntimeInterface(ABC):
    config: AppConfig

    @abstractmethod
    def read(self, action: FileReadAction) -> Observation:
        pass

    @abstractmethod
    def write(self, action: FileWriteAction) -> Observation:
        pass


class FileEditRuntimeMixin(FileEditRuntimeInterface):
    # Most LLMs have output token limit of 4k tokens.
    # This restricts the number of lines we can edit to avoid exceeding the token limit.
    MAX_LINES_TO_EDIT = 600

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        llm_config = self.config.get_llm_config()

        if llm_config.draft_editor is None:
            llm_config.draft_editor = copy.deepcopy(llm_config)

        # manually set the model name for the draft editor LLM to distinguish token costs
        llm_metrics = Metrics(
            model_name='draft_editor:' + llm_config.draft_editor.model
        )
        if llm_config.draft_editor.caching_prompt:
            logger.info(
                'It is not recommended to cache draft editor LLM prompts as it may incur high costs for the same prompt. '
                'Automatically setting caching_prompt=false.'
            )
            llm_config.draft_editor.caching_prompt = False

        self.draft_editor_llm = LLM(llm_config.draft_editor, metrics=llm_metrics)
        logger.info(
            f'[Draft edit functionality] enabled with LLM: {self.draft_editor_llm}'
        )

    def _validate_range(
        self, start: int, end: int, total_lines: int
    ) -> Observation | None:
        # start and end are 1-indexed and inclusive
        if (
            (start < 1 and start != -1)
            or start > total_lines
            or (start > end and end != -1 and start != -1)
        ):
            return ErrorObservation(
                f'Invalid range for editing: start={start}, end={end}, total lines={total_lines}. start must be >= 1 and <={total_lines} (total lines of the edited file), start <= end, or start == -1 (append to the end of the file).'
            )
        if (
            (end < 1 and end != -1)
            or end > total_lines
            or (end < start and start != -1 and end != -1)
        ):
            return ErrorObservation(
                f'Invalid range for editing: start={start}, end={end}, total lines={total_lines}. end must be >= 1 and <= {total_lines} (total lines of the edited file), end >= start, or end == -1 (to edit till the end of the file).'
            )
        return None

    def _get_lint_error(
        self,
        suffix: str,
        old_content: str,
        new_content: str,
        filepath: str,
        diff: str,
    ) -> ErrorObservation | None:
        linter = DefaultLinter()
        # Copy the original file to a temporary file (with the same ext) and lint it
        with tempfile.NamedTemporaryFile(
            suffix=suffix, mode='w+', encoding='utf-8'
        ) as original_file_copy, tempfile.NamedTemporaryFile(
            suffix=suffix, mode='w+', encoding='utf-8'
        ) as updated_file_copy:
            # Lint the original file
            original_file_copy.write(old_content)
            original_file_copy.flush()

            # Lint the updated file
            updated_file_copy.write(new_content)
            updated_file_copy.flush()

            updated_lint_error = linter.lint_file_diff(
                original_file_copy.name, updated_file_copy.name
            )

            if len(updated_lint_error) > 0:
                _obs = FileEditObservation(
                    content=diff,
                    path=filepath,
                    prev_exist=True,
                    old_content=old_content,
                    new_content=new_content,
                )
                error_message = (
                    (
                        f'\n[Linting failed for edited file {filepath}. {len(updated_lint_error)} lint errors found.]\n'
                        '[begin attempted changes]\n'
                        f'{_obs.visualize_diff(change_applied=False)}\n'
                        '[end attempted changes]\n'
                    )
                    + '-' * 40
                    + '\n'
                )
                error_message += '-' * 20 + 'First 5 lint errors' + '-' * 20 + '\n'
                for i, lint_error in enumerate(updated_lint_error[:5]):
                    error_message += f'[begin lint error {i}]\n'
                    error_message += lint_error.visualize().strip() + '\n'
                    error_message += f'[end lint error {i}]\n'
                    error_message += '-' * 40 + '\n'
                return ErrorObservation(error_message)
        return None

    def edit(self, action: FileEditAction) -> Observation:
        obs = self.read(FileReadAction(path=action.path))
        if (
            isinstance(obs, ErrorObservation)
            and 'File not found'.lower() in obs.content.lower()
        ):
            logger.debug(
                f'Agent attempted to edit a file that does not exist. Creating the file. Error msg: {obs.content}'
            )
            # directly write the new content
            obs = self.write(
                FileWriteAction(path=action.path, content=action.content.strip(), thought=action.thought)
            )
            if isinstance(obs, ErrorObservation):
                return obs
            if not isinstance(obs, FileWriteObservation):
                return FatalErrorObservation(
                    f'Fatal Runtime in editing: Expected FileWriteObservation, got {type(obs)}: {str(obs)}'
                )
            return FileEditObservation(
                content=get_diff('', action.content, action.path),
                path=action.path,
                prev_exist=False,
                old_content='',
                new_content=action.content,
            )
        if not isinstance(obs, FileReadObservation):
            return FatalErrorObservation(
                f'Fatal Runtime in editing: Expected FileReadObservation, got {type(obs)}: {str(obs)}'
            )
        original_file_content = obs.content
        old_file_lines = original_file_content.split('\n')
        if action.end != -1 and action.end > len(old_file_lines):
            action.end = len(old_file_lines)
        # NOTE: start and end are 1-indexed
        start = action.start
        end = action.end
        # validate the range
        error = self._validate_range(start, end, len(old_file_lines))
        if error is not None:
            return error

        # append to the end of the file
        if start == -1:
            updated_content = '\n'.join(old_file_lines + action.content.split('\n'))
            diff = get_diff(original_file_content, updated_content, action.path)
            # Lint the updated content
            if self.config.sandbox.enable_auto_lint:
                suffix = os.path.splitext(action.path)[1]

                error_obs = self._get_lint_error(
                    suffix,
                    original_file_content,
                    updated_content,
                    action.path,
                    diff,
                )
                if error_obs is not None:
                    return error_obs

            obs = self.write(FileWriteAction(path=action.path, content=updated_content, thought=action.thought))
            return FileEditObservation(
                content=diff,
                path=action.path,
                prev_exist=True,
                old_content=original_file_content,
                new_content=updated_content,
            )

        # Get the 0-indexed start and end
        start_idx = start - 1
        if end != -1:
            # remove 1 to make it 0-indexed
            # then add 1 since the `end` is inclusive
            end_idx = end - 1 + 1
        else:
            # end == -1 means the user wants to edit till the end of the file
            end_idx = len(old_file_lines)

        # Get the range of lines to edit - reject if too long
        length_of_range = end_idx - start_idx
        if length_of_range > self.MAX_LINES_TO_EDIT + 1:
            logger.info(f'Edit range too long: {length_of_range} lines. Getting new start and end line numbers.')
            try:
                start, end = get_start_end_range(self.draft_editor_llm, action.thought)
            except Exception as e:
                logger.error(f'Failed to get start and end line numbers: {e}')
                start, end = None, None
            logger.info(f'New start and end line numbers: {start}, {end}')
            if isinstance(start, int) and isinstance(end, int):
                start_idx = start if start != -1 else start_idx
                end_idx = end if end != -1 else end_idx
                
        length_of_range = end_idx - start_idx
        if length_of_range > self.MAX_LINES_TO_EDIT + 1:
            error_msg = (
                f'[Edit error: The range of lines to edit is too long.]\n'
                f'[The maximum number of lines allowed to edit at once is {self.MAX_LINES_TO_EDIT}. '
                f'Got (L{start_idx + 1}-L{end_idx}) {length_of_range} lines.]\n'  # [start_idx, end_idx), so no need to + 1
                f'[Please use the start and end line numbers for the edit action to specify the range of lines to edit: `<file_edit path=str start=int end=int>code here</file_edit>`]'
            )
            error_msg += get_relevant_snippets(action, original_file_content, self.MAX_LINES_TO_EDIT)
            return ErrorObservation(error_msg)
        
        # 找到第一个空行作为分界
        new_start_idx = max(0, start_idx - 50) # 902-5 = 897
        while new_start_idx < max(0, start_idx-10):
            if old_file_lines[new_start_idx].strip() == '':
                break
            else:
                new_start_idx += 1
        new_end_idx = min(len(old_file_lines)-1, end_idx + 100) # 920+5 = 925
        while new_end_idx > min(len(old_file_lines)-1, end_idx+20):
            if old_file_lines[new_end_idx].strip() == '':
                break
            else:
                new_end_idx -= 1
                
        start_idx = max(0, new_start_idx)
        end_idx = min(len(old_file_lines)-1, new_end_idx)
        
        content_to_edit = '\n'.join(old_file_lines[start_idx:end_idx])
        self.draft_editor_llm.reset()
        _edited_content = get_new_file_contents(
            self.draft_editor_llm, content_to_edit, action.content, action.thought
        )
        if _edited_content is None:
            ret_err = ErrorObservation(
                'Failed to get new file contents. '
                'Please try to reduce the number of edits and try again.'
            )
            ret_err.llm_metrics = self.draft_editor_llm.metrics
            return ret_err

        # piece the updated content with the unchanged content
        updated_lines = (
            old_file_lines[:start_idx]
            + _edited_content.split('\n')
            + old_file_lines[end_idx:]
        )
        updated_content = '\n'.join(updated_lines)
        diff = get_diff(original_file_content, updated_content, action.path)
        error_msg = ""
        if diff.strip() == '' or "".join(original_file_content.splitlines()) == "".join(updated_content.splitlines()):
            text = "\n".join(old_file_lines[start_idx:end_idx])
            edit_chunk = Chunk(text=text, line_range=(start_idx+1, start_idx+1+len(text.split('\n'))-1))
            error_msg = (
                "The edit did not produce any useful code changes. Please try again.\n"
                "Below is the code snippet you attempted to edit, but no changes were made:\n"
                f"[Begin Snippet: Lines {edit_chunk.line_range[0]}-{edit_chunk.line_range[1]}]\n"
                f"Browse the file around this snippet using: `open_file(\"{action.path}\", {(edit_chunk.line_range[0] + edit_chunk.line_range[1]) // 2})`\n"
                f"{edit_chunk.visualize()}\n"
                "[End Snippet]\n"
                + "-" * 40 + "\n"
                "Please edit the code in the snippet as needed.\n"
                "**IMPORTANT**: Reduce the edit range to fewer than {self.MAX_LINES_TO_EDIT} lines by specifying `start` and `end` in your edit action, "
                "e.g., `<file_edit path=\"{action.path}\" start=[PUT LINE NUMBER HERE] end=[PUT LINE NUMBER HERE] />`."
            )

        # Lint the updated content
        if self.config.sandbox.enable_auto_lint:
            suffix = os.path.splitext(action.path)[1]
            error_obs = self._get_lint_error(
                suffix, original_file_content, updated_content, action.path, diff
            )
            if error_obs is not None:
                error_obs.llm_metrics = self.draft_editor_llm.metrics
                return error_obs

        obs = self.write(FileWriteAction(path=action.path, content=updated_content, thought=action.thought))
        ret_obs = FileEditObservation(
            content=diff,
            path=action.path,
            prev_exist=True,
            old_content=original_file_content,
            new_content=updated_content,
            error_msg=error_msg,
        )
        ret_obs.llm_metrics = self.draft_editor_llm.metrics
        return ret_obs
