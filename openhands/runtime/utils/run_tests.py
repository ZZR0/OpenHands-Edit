import asyncio
import os
import re

from openhands.events.action import RunRegressionAction
from openhands.events.observation import (
    FatalErrorObservation,
    RunRegressionObservation,
)
from openhands.runtime.utils.run_tests_utils import (
    MAP_REPO_TO_PARSER,
    MAP_REPO_TO_TESTCASE_PARSER,
    TestStatus,
)

# Constants - Logging
APPLY_PATCH_FAIL = '>>>>> Patch Apply Failed'
APPLY_PATCH_PASS = '>>>>> Applied Patch'
INSTALL_FAIL = '>>>>> Init Failed'
INSTALL_PASS = '>>>>> Init Succeeded'
INSTALL_TIMEOUT = '>>>>> Init Timed Out'
RESET_FAILED = '>>>>> Reset Failed'
TESTS_ERROR = '>>>>> Tests Errored'
TESTS_FAILED = '>>>>> Some Tests Failed'
TESTS_PASSED = '>>>>> All Tests Passed'
TESTS_TIMEOUT = '>>>>> Tests Timed Out'


def get_logs_eval(content: str, repo: str) -> tuple[dict[str, str], bool]:
    log_parser = MAP_REPO_TO_PARSER[repo]

    if (
        any(
            [
                x in content
                for x in [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                    'Failed to reset task environment',
                ]
            ]
        )
        # or "applied patch" not in content.lower()
    ):
        # Eval patch was not applied successfully
        return {}, False

    # Get status map of evaluation results
    content = content.split(f'{APPLY_PATCH_PASS} (pred)')[-1]
    return log_parser(content), True


def _get_swebench_workspace_dir_name(repo: str, version: str) -> str:
    return f'{repo}__{version}'.replace('/', '__')


class RunTestsSession:
    """A class that maintains a pexpect process and provides a simple interface for running commands and interacting with the shell."""

    def __init__(
        self,
        work_dir: str,
        test_command: str | None = None,
        testcases: list[str] | None = None,
    ):
        self.work_dir = work_dir
        self.test_command = test_command
        self.testcases = testcases

    async def run(
        self, action: RunRegressionAction
    ) -> RunRegressionObservation | FatalErrorObservation:
        try:
            timeout = action.timeout
            repo = action.repo
            test_command = action.test_command or self.test_command
            testcases = action.testcases or self.testcases
            assert (
                timeout is not None
            ), f'Timeout argument is required for RunRegression: {action}'
            assert (
                repo is not None
            ), f'Repo argument is required for RunRegression: {action}'
            assert (
                test_command is not None
            ), f'Test command argument is required for RunRegression: {action}'
            # assert (
            #     testcases is not None
            # ), f'Test cases argument is required for RunRegression: {action}'

            # test_command = " ".join(['/openhands/micromamba/bin/micromamba', 'run', '-n', 'openhands', 'poetry', 'run',]) + ' ' + test_command
            env = os.environ.copy()
            env['PATH'] = (
                '/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/condabin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
            )
            proc = await asyncio.create_subprocess_shell(
                test_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.join(
                    self.work_dir,
                    _get_swebench_workspace_dir_name(repo, action.version),
                ),
                env=env,
            )
            stdout, stderr = await proc.communicate()

            exit_code = proc.returncode or 0
            output = stdout.decode()
            error = stderr.decode()
            report, _ = get_logs_eval(output + '\n' + error, repo)

            def check_report_key_begin_with_test(report: dict[str, str]):
                # report 中要至少有一个 key 以 test_ 开头
                return any(
                    [
                        (x.split('::')[-1]).startswith('test_') or x.startswith('test_')
                        for x in report.keys()
                    ]
                )

            is_legacy = True
            if testcases:
                if 'pytest' in test_command or 'tox' in test_command:
                    if exit_code not in [0, 1] or not check_report_key_begin_with_test(
                        report
                    ):
                        is_legacy = False
                elif repo == 'django/django':
                    if (
                        exit_code not in [0, 1]
                        or not check_report_key_begin_with_test(report)
                        or '(unittest.loader._FailedTest) ... ERROR'
                        in (output + '\n' + error)
                        or not re.search(r'Ran \d+ tests in ', (output + '\n' + error))
                    ):
                        # 必须要存在  'Ran xxx tests in ' 这样的文字
                        is_legacy = False
                else:
                    assert repo == 'sympy/sympy', f'Unknown repo: {repo}'
                    if (
                        exit_code not in [0, 1]
                        or not check_report_key_begin_with_test(report)
                        or not re.search(
                            r'tests finished: \d+ passed', (output + '\n' + error)
                        )
                    ):
                        # 必须要存在 'tests finished: 0 passed' 之类的
                        is_legacy = False
            else:
                is_legacy = exit_code in [0, 1]

            passed = True
            tests_not_passed = []
            if is_legacy:
                allow_skipped = False
                if not testcases:
                    testcases = list(report.keys())
                    allow_skipped = True
                for testcase in testcases:
                    # 如果 1. testcase 不在 report 中 2. testcase 不是 PASSED（如果 allow_skipped 那么 SKIPPED 也算）
                    if testcase not in report or (
                        report[testcase] != TestStatus.PASSED.value
                        and (
                            not allow_skipped
                            or report[testcase] != TestStatus.SKIPPED.value
                        )
                    ):
                        passed = False
                        tests_not_passed.append(testcase)
            else:
                passed = False

            if passed:
                content = 'All regression tests passed.\n'
            else:
                content = 'Some tests failed.\n'
                # content += "Output:\n"
                # content += output
                # if error:
                #     content += "\nError Output:\n"
                #     content += error
            if is_legacy:
                if not passed:
                    if error:
                        content += '------------------- TEST STDERROR BEGIN ---------------------\n'
                        content += error
                        content += '-------------------- TEST STDERROR END ----------------------\n'

                    testcase_output = MAP_REPO_TO_TESTCASE_PARSER[repo](
                        output + '\n' + error, report
                    )
                    for testcase, err_output in testcase_output.items():
                        if testcase not in tests_not_passed:
                            continue
                        status = report[testcase]
                        content += (
                            '------------------- TESTCASE BEGIN ---------------------\n'
                        )
                        content += f'\nTestcase: {testcase}\n'
                        content += f'Status: {status}\n'
                        content += f'Error output:\n{err_output}\n'
                        content += (
                            '-------------------- TESTCASE END ----------------------\n'
                        )

                    other_testcases = set(tests_not_passed) - set(
                        testcase_output.keys()
                    )
                    if other_testcases:
                        # content += '------------------- TESTCASE  ---------------------\n'
                        content += '\nOther testcases not passed:\n'
                        for testcase in other_testcases:
                            status = report.get(testcase, 'MISSED')
                            content += f'Testcase: {testcase}, Status: {status}\n'
            else:
                content += 'The whole test output is below:\n'
                if error:
                    content += '------------------- TEST STDERROR BEGIN ---------------------\n'
                    content += error
                    content += '-------------------- TEST STDERROR END ----------------------\n'
                content += (
                    '------------------- TEST OUTPUT BEGIN ---------------------\n'
                )
                content += output
                content += (
                    '-------------------- TEST OUTPUT END ----------------------\n'
                )

            # content += f"\nDebug: test_command: {test_command}"
            # content += f"\nDebug: env: {env}"
            # content += f"\nDebug: work_dir: {self.work_dir}"
            # content += f"\nexit_code: {exit_code}"
            # import json
            # content += f"\nDebug: report: {json.dumps(report, indent=4)}"
            # content += f"\nDebug: repo: {repo}"
            # # content += f"\nDebug: get_logs_eval: {get_logs_eval(output, repo)}"
            # content += f"\nDebug: output: {output}"
            # content += f"\nDebug: testcases: {testcases}"
            # # content += f"\nDebug: report: {report}"
            # content += f"\nDebug: passed: {passed}"
            # content += f"\nDebug: tests_not_passed: {tests_not_passed}"
            # if not passed:
            #     content += f"\nDebug: testcase_output: {testcase_output}"
            return RunRegressionObservation(
                content=content,
                test_command=test_command,
                exit_code=0 if is_legacy else exit_code,
                passed=passed,
                report=report,
                output=output,
                error_output=error,
                tests_not_passed=tests_not_passed,
                testcase_output=testcase_output if not passed and is_legacy else None,
            )
        except UnicodeDecodeError as e:
            return FatalErrorObservation(
                f'Run regression failed: Command output could not be decoded as utf-8. {str(e)}'
            )
