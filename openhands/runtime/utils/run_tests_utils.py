import re
from enum import Enum

# from swebench.harness.constants import TestStatus


class TestStatus(Enum):
    FAILED = 'FAILED'
    PASSED = 'PASSED'
    SKIPPED = 'SKIPPED'
    ERROR = 'ERROR'


def parse_log_pytest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split('\n'):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(' - ', ' ')
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_options(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework with options

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    option_pattern = re.compile(r'(.*?)\[(.*)\]')
    test_status_map = {}
    for line in log.split('\n'):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(' - ', ' ')
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            has_option = option_pattern.search(test_case[1])
            if has_option:
                main, option = has_option.groups()
                if (
                    option.startswith('/')
                    and not option.startswith('//')
                    and '*' not in option
                ):
                    option = '/' + option.split('/')[-1]
                test_name = f'{main}[{option}]'
            else:
                test_name = test_case[1]
            test_status_map[test_name] = test_case[0]
    return test_status_map


def parse_log_django(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Django tester framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    lines = log.split('\n')

    prev_test = None
    for line in lines:
        line = line.strip()

        # This isn't ideal but the test output spans multiple lines
        if '--version is equivalent to version' in line:
            test_status_map['--version is equivalent to version'] = (
                TestStatus.PASSED.value
            )

        if line.count(' ... ') > 1:
            sublines = line.split(' ... ')
            for subline in sublines[:-2]:
                test = subline.strip()
                test_status_map[test] = TestStatus.FAILED.value
            line = ' ... '.join(sublines[-2:])

        # Log it in case of error
        if ' ... ' in line:
            prev_test = line.split(' ... ')[0]

        pass_suffixes = (' ... ok', ' ... OK', ' ...  OK')
        for suffix in pass_suffixes:
            if line.endswith(suffix):
                # TODO: Temporary, exclusive fix for django__django-7188
                # The proper fix should involve somehow getting the test results to
                # print on a separate line, rather than the same line
                if line.strip().startswith(
                    'Applying sites.0002_alter_domain_unique...test_no_migrations'
                ):
                    line = line.split('...', 1)[-1].strip()
                test = line.rsplit(suffix, 1)[0]
                test_status_map[test] = TestStatus.PASSED.value
                break
        if ' ... skipped' in line:
            test = line.split(' ... skipped')[0]
            test_status_map[test] = TestStatus.SKIPPED.value
        if line.endswith(' ... FAIL'):
            test = line.split(' ... FAIL')[0]
            test_status_map[test] = TestStatus.FAILED.value
        if line.startswith('FAIL:'):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.FAILED.value
        if line.endswith(' ... ERROR'):
            test = line.split(' ... ERROR')[0]
            test_status_map[test] = TestStatus.ERROR.value
        if line.startswith('ERROR:'):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.ERROR.value

        if line.lstrip().startswith('ok') and prev_test is not None:
            # It means the test passed, but there's some additional output (including new lines)
            # between "..." and "ok" message
            test = prev_test
            test_status_map[test] = TestStatus.PASSED.value

    # TODO: This is very brittle, we should do better
    # There's a bug in the django logger, such that sometimes a test output near the end gets
    # interrupted by a particular long multiline print statement.
    # We have observed this in one of 3 forms:
    # - "{test_name} ... Testing against Django installed in {*} silenced.\nok"
    # - "{test_name} ... Internal Server Error: \/(.*)\/\nok"
    # - "{test_name} ... System check identified no issues (0 silenced).\nok"
    patterns = [
        r'^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$',
        r'^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$',
        r'^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, log, re.MULTILINE):
            test_name = match.group(1)
            test_status_map[test_name] = TestStatus.PASSED.value
    return test_status_map


def parse_log_pytest_v2(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework (Later Version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = ''.join([chr(char) for char in range(1, 32)])
    for line in log.split('\n'):
        line = re.sub(r'\[(\d+)m', '', line)
        translator = str.maketrans('', '', escapes)
        line = line.translate(translator)
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(' - ', ' ')
            test_case = line.split()
            test_status_map[test_case[1]] = test_case[0]
        # Support older pytest versions by checking if the line ends with the test status
        elif any([line.endswith(x.value) for x in TestStatus]):
            test_case = line.split()
            test_status_map[test_case[0]] = test_case[1]
    return test_status_map


def parse_log_seaborn(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with seaborn testing framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split('\n'):
        if line.startswith(TestStatus.FAILED.value):
            test_case = line.split()[1]
            test_status_map[test_case] = TestStatus.FAILED.value
        elif f' {TestStatus.PASSED.value} ' in line:
            parts = line.split()
            if parts[1] == TestStatus.PASSED.value:
                test_case = parts[0]
                test_status_map[test_case] = TestStatus.PASSED.value
        elif line.startswith(TestStatus.PASSED.value):
            parts = line.split()
            test_case = parts[1]
            test_status_map[test_case] = TestStatus.PASSED.value
    return test_status_map


def parse_log_sympy(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Sympy framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    pattern = r'(_*) (.*)\.py:(.*) (_*)'
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f'{match[1]}.py:{match[2]}'
        test_status_map[test_case] = TestStatus.FAILED.value
    for line in log.split('\n'):
        line = line.strip()
        if line.startswith('test_'):
            if line.endswith(' E'):
                test = line.split()[0]
                test_status_map[test] = TestStatus.ERROR.value
            if line.endswith(' F'):
                test = line.split()[0]
                test_status_map[test] = TestStatus.FAILED.value
            if line.endswith(' ok'):
                test = line.split()[0]
                test_status_map[test] = TestStatus.PASSED.value
    return test_status_map


def parse_log_matplotlib(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split('\n'):
        line = line.replace('MouseButton.LEFT', '1')
        line = line.replace('MouseButton.RIGHT', '3')
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(' - ', ' ')
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


parse_log_astroid = parse_log_pytest
parse_log_flask = parse_log_pytest
parse_log_marshmallow = parse_log_pytest
parse_log_pvlib = parse_log_pytest
parse_log_pyvista = parse_log_pytest
parse_log_sqlfluff = parse_log_pytest
parse_log_xarray = parse_log_pytest

parse_log_pydicom = parse_log_pytest_options
parse_log_requests = parse_log_pytest_options
parse_log_pylint = parse_log_pytest_options

parse_log_astropy = parse_log_pytest_v2
parse_log_scikit = parse_log_pytest_v2
parse_log_sphinx = parse_log_pytest_v2


MAP_REPO_TO_PARSER = {
    'astropy/astropy': parse_log_astropy,
    'django/django': parse_log_django,
    'marshmallow-code/marshmallow': parse_log_marshmallow,
    'matplotlib/matplotlib': parse_log_matplotlib,
    'mwaskom/seaborn': parse_log_seaborn,
    'pallets/flask': parse_log_flask,
    'psf/requests': parse_log_requests,
    'pvlib/pvlib-python': parse_log_pvlib,
    'pydata/xarray': parse_log_xarray,
    'pydicom/pydicom': parse_log_pydicom,
    'pylint-dev/astroid': parse_log_astroid,
    'pylint-dev/pylint': parse_log_pylint,
    'pytest-dev/pytest': parse_log_pytest,
    'pyvista/pyvista': parse_log_pyvista,
    'scikit-learn/scikit-learn': parse_log_scikit,
    'sqlfluff/sqlfluff': parse_log_sqlfluff,
    'sphinx-doc/sphinx': parse_log_sphinx,
    'sympy/sympy': parse_log_sympy,
}


def parse_testcase_log_pytest(log: str, report: dict[str, str]):
    FAILURE_BEGIN = '= FAILURES ='
    PASSES_BEGIN = '= PASSES ='
    ERRORS_BEGIN = '= ERRORS ='
    WARNING_SUMMARY = '= warnings summary ='
    CAPTUREDS = [
        '- Captured stdout call -',
        '- Captured stderr call -',
        '- Captured log call -',
        '- Captured stdout teardown -',
        '- Captured stderr teardown -',
        '- Captured log teardown -',
        WARNING_SUMMARY,
    ]

    SUMMARY = '= short test summary info ='
    log_lines = log.split('\n')
    # 根据行内是否出现 FAILURE_BEGIN, PASSES_BEGIN 和 SUMMARY 来切分 log_lines

    summary_index = (
        [i for i, line in enumerate(log_lines) if SUMMARY in line] or [len(log_lines)]
    )[-1]
    # 感觉 summary 的话取最后一个比较合适一点（应对 pytest 的那些 case）
    # failure 和 pass 应该还得是第一个
    # 不过即便取了第一个感觉对于 pytest 可能还是会有问题，我明明加了 -s 参数的吧
    failure_index = (
        [i for i, line in enumerate(log_lines) if FAILURE_BEGIN in line]
        or [summary_index]
    )[0]
    passes_index = (
        [i for i, line in enumerate(log_lines) if PASSES_BEGIN in line]
        or [summary_index]
    )[0]
    error_index = (
        [i for i, line in enumerate(log_lines) if ERRORS_BEGIN in line]
        or [summary_index]
    )[0]

    # 根据这些 index 获得不同的部分，各个部分之间顺序不固定
    # get_end_line = lambda x, ls: min(le for le in ls if le >= x)
    def get_end_line(x, ls):
        return min(le for le in ls if le >= x)

    failure_lines = log_lines[
        failure_index : get_end_line(
            failure_index, [passes_index, error_index, summary_index]
        )
    ]
    # passes_lines = log_lines[
    #     passes_index : get_end_line(
    #         passes_index, [failure_index, error_index, summary_index]
    #     )
    # ]
    error_lines = log_lines[
        error_index : get_end_line(
            error_index, [failure_index, passes_index, summary_index]
        )
    ]

    failed_case_begin_lines: list[tuple[int, str | None]] = []
    for case, res in report.items():
        case_base = '.'.join(case.split('::')[1:])
        if res != TestStatus.FAILED.value:
            continue
        exist_lines = [
            i
            for i, line in enumerate(failure_lines)
            if '__ ' + case_base + ' __' in line
        ]
        if not exist_lines:
            continue
        failed_case_begin_lines.append((exist_lines[0], case))

    failed_case_begin_lines.sort()
    failed_case_begin_lines.append((len(failure_lines), None))

    result = {}
    for i in range(len(failed_case_begin_lines) - 1):
        case_begin = failed_case_begin_lines[i][0]
        case_end = failed_case_begin_lines[i + 1][0]
        case_name = failed_case_begin_lines[i][1]
        assert case_name
        case_lines = failure_lines[case_begin:case_end]
        # 截取到 CAPTURED_STDOUT 或者 CAPTURED_STDERR 或者结尾之前
        captured_lines = [
            i
            for i, line in enumerate(case_lines)
            if any(captured in line for captured in CAPTUREDS)
        ]
        captured_lines.append(len(case_lines))
        case_lines = case_lines[1 : captured_lines[0]]
        result[case_name] = '\n'.join(case_lines)

    error_case_begin_lines: list[tuple[int, str | None]] = []
    for case, res in report.items():
        case_base = '.'.join(case.split('::')[1:])
        if res != TestStatus.ERROR.value:
            continue
        exist_lines = [
            i
            for i, line in enumerate(error_lines)
            if '_ ERROR' in line and case_base + ' _' in line
        ]
        if not exist_lines:
            continue
        error_case_begin_lines.append((exist_lines[0], case))

    error_case_begin_lines.sort()
    error_case_begin_lines.append((len(error_lines), None))

    for i in range(len(error_case_begin_lines) - 1):
        case_begin = error_case_begin_lines[i][0]
        case_end = error_case_begin_lines[i + 1][0]
        case_name = error_case_begin_lines[i][1]
        assert case_name
        case_lines = error_lines[case_begin:case_end]
        # 截取到 CAPTURED_STDOUT 或者 CAPTURED_STDERR 或者结尾之前
        captured_lines = [
            i
            for i, line in enumerate(case_lines)
            if any(captured in line for captured in CAPTUREDS)
        ]
        captured_lines.append(len(case_lines))
        case_lines = case_lines[1 : captured_lines[0]]
        result[case_name] = '\n'.join(case_lines)

    return result


def parse_testcase_log_django(log: str, report: dict[str, str]):
    TESTCASES_DIVIDE_LINE = '==============='
    TESTCASES_BEGIN_LINE = '---------------'

    result = {}
    log_lines = log.split('\n')
    for case, res in report.items():
        if res == TestStatus.FAILED.value or res == TestStatus.ERROR.value:
            for prefix in ['FAIL', 'ERROR']:
                pattern = prefix + ': ' + case
                if pattern not in log:
                    continue
                if log.count(pattern) == 1:
                    content = log.split(pattern)[-1].split(TESTCASES_DIVIDE_LINE)[0]
                    content_lines = content.split('\n')
                    begin_line = (
                        [
                            i + 1
                            for i, line in enumerate(content_lines)
                            if TESTCASES_BEGIN_LINE in line
                        ]
                        or [0]
                    )[0]
                    content = '\n'.join(content_lines[begin_line:])
                    result[case] = content
                else:
                    # 找到每一个出现 pattern 的行
                    lines = [i for i, line in enumerate(log_lines) if pattern in line]
                    content = ''
                    for line in lines:
                        # 找到 line 之后第一个包含 TESTCASES_DIVIDE_LINE 的行
                        next_line = (
                            [
                                i + line + 1
                                for i, line_content in enumerate(log_lines[line:])
                                if TESTCASES_DIVIDE_LINE in line_content
                            ]
                            or [len(log_lines)]
                        )[0]
                        content += '\n'.join(log_lines[line:next_line]) + '\n'
                    result[case] = content

    return result


def parse_testcase_log_sympy(log: str, report: dict[str, str]):
    FINISHED = 'tests finished: '
    DIVIDE_LINE = '______________________________________'

    log_lines = log.split('\n')
    result = {}
    for case, res in report.items():
        if res == TestStatus.FAILED.value or res == TestStatus.ERROR.value:
            pattern1 = '__ '
            pattern2 = case + ' __'
            # 找到同时出现两个 pattern 的行，然后向下找到出现 DIVIDE_LINE 或者 FINISHED 的行
            lines = [
                i
                for i, line in enumerate(log_lines)
                if pattern1 in line and pattern2 in line
            ]
            if not lines:
                continue
            begin_line = lines[0] + 1
            # 然后向下找到出现 DIVIDE_LINE 或者 FINISHED 的行
            end_line = (
                [
                    i + begin_line
                    for i, line in enumerate(log_lines[begin_line:])
                    if DIVIDE_LINE in line or FINISHED in line
                ]
                or [len(log_lines)]
            )[0]
            content = '\n'.join(log_lines[begin_line:end_line])
            result[case] = content

    return result


MAP_REPO_TO_TESTCASE_PARSER = {
    'astropy/astropy': parse_testcase_log_pytest,
    'django/django': parse_testcase_log_django,
    'marshmallow-code/marshmallow': parse_testcase_log_pytest,
    'matplotlib/matplotlib': parse_testcase_log_pytest,
    'mwaskom/seaborn': parse_testcase_log_pytest,
    'pallets/flask': parse_testcase_log_pytest,
    'psf/requests': parse_testcase_log_pytest,
    'pvlib/pvlib-python': parse_testcase_log_pytest,
    'pydata/xarray': parse_testcase_log_pytest,
    'pydicom/pydicom': parse_testcase_log_pytest,
    'pylint-dev/astroid': parse_testcase_log_pytest,
    'pylint-dev/pylint': parse_testcase_log_pytest,
    'pytest-dev/pytest': parse_testcase_log_pytest,
    'pyvista/pyvista': parse_testcase_log_pytest,
    'scikit-learn/scikit-learn': parse_testcase_log_pytest,
    'sqlfluff/sqlfluff': parse_testcase_log_pytest,
    'sphinx-doc/sphinx': parse_testcase_log_pytest,
    'sympy/sympy': parse_testcase_log_sympy,
}
