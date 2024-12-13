import json

import requests
from openai import OpenAI
from tqdm import tqdm


def test_gemini_api():
    # data_path = '/hdd2/zzr/OpenHands-fn-calling/data.json'
    data_path = '/hdd2/zzr/OpenHands-fn-calling/err.json'
    data = json.load(open(data_path, 'r'))
    headers = {'Content-Type': 'application/json'}
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=AIzaSyAFI9-SuuRLPCmCDuexBpr_T6uf6S_i1nk'
    print(data.keys())
    # import pdb; pdb.set_trace()
    # data['contents'] = data['contents'][:-1]
    for c in data['contents']:
        print(c['role'])
        print(len(c['parts']))
    print(data['contents'][-3]['parts'][0].keys())
    print(data['contents'][-1]['parts'][0]['text'])
    data['contents'] = data['contents'][1:]
    data['contents'] = (
        data['contents']
        + [data['contents'][-2], data['contents'][-1]]
        + [data['contents'][-2], data['contents'][-1]]
    )
    # data['contents'][-1]["parts"][0]['text'] = data['contents'][-1]["parts"][0]['text'][:1081]
    # print(data['contents'][-1]["parts"][0]['text'])
    # print(data['contents'][-3]['role'])
    # print(len(data['contents'][-3]['parts']))
    # print(data['contents'][-3]['parts'][0].keys())
    # print(data['contents'][-1]['role'])
    # print(len(data['contents'][-1]['parts']))
    # print(data['contents'][-1]['parts'])
    # print(data['contents'][-1]['parts'][0].keys())
    # print(data['contents'][-1]['parts'][1].keys())
    response = requests.post(url, headers=headers, json=data)
    print(response.json())


def test_litellm():
    from litellm import completion

    # message_path = '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-1.5-flash_maxiter_100_N_v2.1-no-hint-nofc_notest-run_1/llm_completions/astropy__astropy-7606/draft_editor:gemini__gemini-1.5-flash-1734070124.4145105.json'
    message_path = '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgentEdit/gemini-2.0-flash-exp_maxiter_100_N_v2.1-no-hint-nofc_wtest-run_1/llm_completions/django__django-11265/gemini__gemini-2.0-flash-exp-1734099381.7370067.json'
    with open(message_path, 'r') as f:
        data = json.load(f)
    messages = data['messages']
    # tools = data["kwargs"]["tools"]
    # messages = [
    #     {
    #         'content': "You are a helpful assistant that can interact with a computer to solve tasks.\n<IMPORTANT>\n* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.\n</IMPORTANT>\n",
    #         'role': 'system'
    #     },
    #     {
    #         'content':  "<uploaded_files>\n/workspace/django__django__4.2\n</uploaded_files>\nI've uploaded a python code repository in the directory django__django__4.2. Consider the following PR description:\n\n<pr_description>\nUserCreationForm should save data from ManyToMany form fields\nDescription\n\t\nWhen using contrib.auth.forms.UserCreationForm with a custom User model which has ManyToManyField fields, the data in all related form fields (e.g. a ModelMultipleChoiceField) is not saved. \nThis is because unlike its parent class django.forms.ModelForm, UserCreationForm.save(commit=True) omits to call self.save_m2m(). \nThis has been discussed on the #django-developers mailing list \u200bhttps://groups.google.com/u/1/g/django-developers/c/2jj-ecoBwE4 and I'm ready to work on a PR.\n\n</pr_description>\n\nCan you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\nI've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\nYour task is to make the minimal changes to non-tests files in the /workspace directory to ensure the <pr_description> is satisfied.\n### Follow These Steps to Resolve the Issue:\n1. **Familiarize Yourself with the Repository**:\n   - Explore the codebase to understand its structure and identify relevant files, classes, functions, or variables that may be affected by the `<pr_description>`.\n2. **Analyze the Problem**:\n   - Identify the specific areas of the codebase that require changes.\n   - Provide a detailed breakdown of the files, code locations, and any related dependencies that need to be addressed.\n3. **Implement the Fix**:\n   - Edit the source code in the identified locations to resolve the issue.\n   - Ensure that your changes are efficient, clean, and adhere to Python best practices.\n4. **Handle Edge Cases**:\n   - Consider potential edge cases and ensure your solution is robust enough to handle them.\n5. **Rerun Your Patch**:\n   - After making the necessary changes, rerun your patch to verify its functionality. Note that you do not need to run any tests yourself; the testing process will be handled by someone else. Once you have completed your changes, simply return it.\n\n### Additional Notes:\n   - Be thorough in your analysis and implementation. It’s okay if your response is detailed and lengthy, as long as it fully addresses the problem.\n   - Clearly document your reasoning, approach, and the changes made to the codebase.\n",
    #         'role': 'user'
    #     },
    #     {
    #         'content': [],
    #         'role': 'assistant',
    #         'tool_calls': [
    #             {
    #                 'index': 0,
    #                 'function': {
    #                     'arguments': '{"command":"ls /workspace/django__django__4.2"}',
    #                     'name': 'execute_bash'
    #                 },
    #                 'id': 'call_0NMUDgSnYkPlW79Mn113FcVi',
    #                 'type': 'function'
    #             }
    #         ]
    #     },
    #     {
    #         'content': 'OBSERVATION:\nls /workspace/django__django__4.2\r\nAUTHORS\t\t  INSTALL\t  README.rst  js_tests\t      setup.cfg\r\nCONTRIBUTING.rst  LICENSE\t  django      package.json    setup.py\r\nDjango.egg-info   LICENSE.python  docs\t      pyproject.toml  tests\r\nGruntfile.js\t  MANIFEST.in\t  extras      scripts\t      tox.ini[Python Interpreter: /opt/miniconda3/envs/testbed/bin/python]\nroot@openhands-workspace:/workspace/django__django__4.2 # \n[Command finished with exit code 0]',
    #         'role': 'tool',
    #         'cache_control': {
    #             'type': 'ephemeral'
    #         },
    #         'tool_call_id': 'call_0NMUDgSnYkPlW79Mn113FcVi',
    #         'name': 'execute_bash'
    #     }
    # ]
    # messages=[
    #     {'role': 'system', 'content': 'You are a helpful assistant.'},
    #     # {'role': 'user', 'content': "What's the weather like in Beijing?"},
    #     {'role': 'user', 'content': "Hello, please introduce gemini-2.0-flash for me."},
    # ]
    # tools = [{'type': 'function', 'function': {'name': 'execute_bash', 'description': 'Execute a bash command in the terminal.\n* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.\n* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.\n* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.\n', 'parameters': {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.'}}, 'required': ['command']}}}, {'type': 'function', 'function': {'name': 'finish', 'description': 'Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.'}}, {'type': 'function', 'function': {'name': 'str_replace_editor', 'description': 'Custom editing tool for viewing, creating and editing files\n* State is persistent across command calls and discussions with the user\n* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n* The `create` command cannot be used if the specified `path` already exists as a file\n* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`\n* The `undo_edit` command will revert the last edit made to the file at `path`\n\nNotes for using the `str_replace` command:\n* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!\n* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique\n* The `new_str` parameter should contain the edited lines that should replace the `old_str`\n', 'parameters': {'type': 'object', 'properties': {'command': {'description': 'The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.', 'enum': ['view', 'create', 'str_replace', 'insert', 'undo_edit'], 'type': 'string'}, 'path': {'description': 'Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.', 'type': 'string'}, 'file_text': {'description': 'Required parameter of `create` command, with the content of the file to be created.', 'type': 'string'}, 'old_str': {'description': 'Required parameter of `str_replace` command containing the string in `path` to replace.', 'type': 'string'}, 'new_str': {'description': 'Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.', 'type': 'string'}, 'insert_line': {'description': 'Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.', 'type': 'integer'}, 'view_range': {'description': 'Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.', 'items': {'type': 'integer'}, 'type': 'array'}}, 'required': ['command', 'path']}}}]
    # re
    # api_key = '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'
    # client = OpenAI(api_key=api_key, base_url='https://openrouter.ai/api/v1')

    # api_key = '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'
    # client = OpenAI(api_key=api_key)

    api_key = 'AIzaSyAFI9-SuuRLPCmCDuexBpr_T6uf6S_i1nk'  # type: ignore
    # base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'  # type: ignore
    # client = OpenAI(api_key=api_key, base_url=base_url)
    # response = completion(
    #     model="gemini-1.5-flash",
    #     api_key=api_key,
    #     base_url=base_url,
    #     messages=messages,
    #     tools=tools
    # )

    # os.environ['GEMINI_API_KEY'] = api_key
    # import pdb; pdb.set_trace()
    response = completion(
        api_key=api_key,
        # model="gemini/gemini-2.0-flash-exp",
        model='gemini/gemini-1.5-flash',
        messages=messages,
        # tools=tools,
        temperature=0.2,
    )

    # response = client.chat.completions.create(
    #     model="gemini-1.5-flash",
    #     messages=messages,
    #     # tools=tools
    # )
    print(response)
    # import pdb; pdb.set_trace()
    print('Finished')


def test_openrouter():
    import json

    import requests

    OPENROUTER_API_KEY = (
        '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'
    )
    response = requests.post(
        url='https://openrouter.ai/api/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        },
        data=json.dumps(
            {
                # "model": "openai/gpt-3.5-turbo", # Optional
                'model': 'openai/gpt-4o-2024-11-20',
                # "model": "google/gemini-2.0-flash-exp:free",
                # "model": "google/gemini-exp-1206:free",
                'messages': [
                    {'role': 'user', 'content': 'What is the meaning of life?'}
                ],
            }
        ),
    )
    print(response.json())


def test_gemini():
    from google import genai

    # Replace the `project` and `location` values with appropriate values for
    # your project.
    client = genai.Client(api_key='AIzaSyAFI9-SuuRLPCmCDuexBpr_T6uf6S_i1nk')
    response = client.models.generate_content(
        model='gemini-2.0-flash-exp', contents='How does AI work?'
    )
    # AIzaSyAFI9-SuuRLPCmCDuexBpr_T6uf6S_i1nk
    print(response.text)


def test_openai():
    # api_key = 'sk-isa0ZCsGSRizprusXPMdwIj1LstdPxYtjmkjMJHAtIqRL7zi'
    # api_key = 'sk-M3bKoucx7o9TSEkK0eC5249f951644E69d880eC269Df061d'
    # client = OpenAI(api_key=api_key, base_url='https://api2.aigcbest.top/v1')

    api_key = (
        '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'
    )
    client = OpenAI(api_key=api_key, base_url='https://openrouter.ai/api/v1')

    # api_key = 'AIzaSyAFI9-SuuRLPCmCDuexBpr_T6uf6S_i1nk'
    # client = OpenAI(api_key=api_key, base_url='https://generativelanguage.googleapis.com/v1beta/openai/')

    # 定义函数
    def get_weather(location):
        return {'Beijing': '晴天'}

    # 定义函数调用
    response = client.chat.completions.create(
        # model="gpt-4o-2024-05-13",
        # model="google/gemini-2.0-flash-exp:free",
        # model="google/gemini-exp-1206:free",
        model='openai/gpt-4o-mini-2024-07-18',
        # model="openai/gpt-4o-2024-11-20",
        # model="gemini-2.0-flash-exp",
        # model="gpt-3.5-turbo",
        # model='claude-3-5-sonnet-20241022',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': "What's the weather like in Beijing?"},
            # {'role': 'user', 'content': "Hello, what's your name? what can you do?"},
        ],
        n=1,
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'description': 'Get the current weather information for a given location',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'location': {
                                'type': 'string',
                                'description': 'The name of the city to get the weather for',
                            }
                        },
                        'required': ['location'],
                    },
                },
            }
        ],
        tool_choice='auto',
    )

    # 打印函数调用结果
    print(response)


def test_anthropic():
    api_key = 'sk-isa0ZCsGSRizprusXPMdwIj1LstdPxYtjmkjMJHAtIqRL7zi'
    # api_key = 'sk-M3bKoucx7o9TSEkK0eC5249f951644E69d880eC269Df061d'

    import http.client
    import json

    conn = http.client.HTTPSConnection('api2.aigcbest.top')
    payload = json.dumps(
        {
            # "model": "c-3-5-sonnet-20241022",
            'model': 'gpt-3.5-turbo',
            # "model": "claude-3-5-sonnet-20241022",
            'max_tokens': 1024,
            'tools': [
                {
                    'name': 'get_weather',
                    'description': 'Get the current weather in a given location',
                    'input_schema': {
                        'type': 'object',
                        'properties': {
                            'location': {
                                'type': 'string',
                                'description': 'The city and state, e.g. San Francisco, CA',
                            }
                        },
                        'required': ['location'],
                    },
                }
            ],
            'messages': [
                {
                    'role': 'user',
                    'content': 'What is the weather like in San Francisco?',
                }
            ],
            'n': 4,
        }
    )
    headers = {
        'Accept': 'application/json',
        'x-api-key': api_key,
        'Content-Type': 'application/json',
    }
    conn.request('POST', '/v1/messages', payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode('utf-8'))


def test_anthropic_2():
    api_key = 'sk-isa0ZCsGSRizprusXPMdwIj1LstdPxYtjmkjMJHAtIqRL7zi'
    # api_key = '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'
    import anthropic

    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=api_key,
        base_url='https://api2.aigcbest.top',
    )
    message = client.messages.create(
        model='claude-3-5-sonnet-20241022',
        max_tokens=1024,
        messages=[
            {'role': 'user', 'content': 'What is the weather like in San Francisco?'}
        ],
        tools=[
            {
                'name': 'get_weather',
                'description': 'Get the current weather in a given location',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'The city and state, e.g. San Francisco, CA',
                        }
                    },
                    'required': ['location'],
                },
            }
        ],
    )
    print(message)


def test_openai2():
    # api_key = 'sk-isa0ZCsGSRizprusXPMdwIj1LstdPxYtjmkjMJHAtIqRL7zi'
    api_key = 'sk-M3bKoucx7o9TSEkK0eC5249f951644E69d880eC269Df061d'
    client = OpenAI(api_key=api_key, base_url='https://api2.aigcbest.top/v1')

    # api_key = 'hk-4qt48b100004811414c08bcb08e49f7a42b44edde99c63e1'
    # client = OpenAI(api_key=api_key, base_url='https://api.openai-hk.com/v1')

    # api_key = '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'
    # client = OpenAI(api_key=api_key)

    # api_key = 'sk-e34104606d4e4c7893466654fbc3524b'
    # client = OpenAI(api_key=api_key, base_url='https://api.deepseek.com/v1')

    # json_path = '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-4o-2024-05-13_maxiter_100_N_v2.1-no-hint-run_1/llm_completions/astropy__astropy-14995/gpt-4o-2024-05-13-1733418790.095674.json'
    # json_path = '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-4o-2024-05-13_maxiter_100_N_v2.1-no-hint-run_1/llm_completions/astropy__astropy-14995/gpt-4o-2024-05-13-1733418784.0715392.json'
    json_path = '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-4o-2024-05-13_maxiter_100_N_v2.1-no-hint-run_1/llm_completions/astropy__astropy-14995/gpt-4o-2024-05-13-1733418787.0466425.json'
    # tools = [
    #     {
    #         'type': 'function',
    #         'function': {
    #             'name': 'get_weather',
    #             'description': 'Get weather of an location, the user shoud supply a location first',
    #             'parameters': {
    #                 'type': 'object',
    #                 'properties': {
    #                     'location': {
    #                         'type': 'string',
    #                         'description': 'The city and state, e.g. San Francisco, CA',
    #                     }
    #                 },
    #                 'required': ['location'],
    #             },
    #         },
    #     },
    # ]

    with open(json_path, 'r') as f:
        data = json.load(f)
    # print(data)
    none_count = 0
    tokens = []
    msgs = []
    for msg in data['messages']:
        # if msg['content'] == []:
        #     msg['content'] = ''
        # else:
        #     msg['content'] = msg['content'][0]['text']
        msgs.append(msg)
    # print(data['messages'][2])
    for i in tqdm(range(20)):
        response = client.chat.completions.create(
            # model="deepseek-chat",
            model='gpt-4o-2024-05-13',
            messages=msgs,
            tools=data['kwargs']['tools'],
            temperature=0.0,
        )
        # response = client.chat.completions.create(
        #     model="deepseek-chat",
        #     messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "How's the weather in Hangzhou?"}],
        #     tools=data['kwargs']['tools'] + tools,
        #     temperature=0.0
        # )
        print(response)
        if response.choices[0].message.content is None:
            none_count += 1
        tokens.append(response.usage.completion_tokens)
        print(response.usage.completion_tokens)
    print(none_count / 1)
    print(sum(tokens) / len(tokens))
    print(tokens)
    # ChatCompletion(id='chatcmpl-AbqKpL0B1IBguoS5FKAXnFUITVtu1', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_700myEDj7dEBrdYGkY3ddT9D', function=Function(arguments='{"path":"/workspace/astropy__astropy__5.2/reproduce_error.py","file_text":"import numpy as np\\nfrom astropy.nddata import NDDataRef\\n\\narray = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])\\nmask = np.array([[0, 1, 64], [8, 0, 1], [2, 1, 0]])\\n\\nnref_nomask = NDDataRef(array)\\nnref_mask = NDDataRef(array, mask=mask)\\n\\n# Attempt to multiply no mask by constant (no mask * no mask)\\ntry:\\n    result = nref_nomask.multiply(1., handle_mask=np.bitwise_or).mask\\n    print(\\"Result of no mask * constant:\\", result)\\nexcept Exception as e:\\n    print(\\"Error during no mask * constant:\\", e)\\n\\n# Attempt to multiply no mask by itself (no mask * no mask)\\ntry:\\n    result = nref_nomask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask\\n    print(\\"Result of no mask * no mask:\\", result)\\nexcept Exception as e:\\n    print(\\"Error during no mask * no mask:\\", e)\\n\\n# Attempt to multiply mask by constant (mask * no mask)\\ntry:\\n    result = nref_mask.multiply(1., handle_mask=np.bitwise_or).mask\\n    print(\\"Result of mask * constant:\\", result)\\nexcept Exception as e:\\n    print(\\"Error during mask * constant:\\", e)\\n\\n# Attempt to multiply mask by itself (mask * mask)\\ntry:\\n    result = nref_mask.multiply(nref_mask, handle_mask=np.bitwise_or).mask\\n    print(\\"Result of mask * mask:\\", result)\\nexcept Exception as e:\\n    print(\\"Error during mask * mask:\\", e)\\n\\n# Attempt to multiply mask by no mask (mask * no mask)\\ntry:\\n    result = nref_mask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask\\n    print(\\"Result of mask * no mask:\\", result)\\nexcept Exception as e:\\n    print(\\"Error during mask * no mask:\\", e)"}', name='create_file'), type='function')]))], created=1733582675, model='gpt-4o-2024-05-13', object='chat.completion', service_tier=None, system_fingerprint='fp_04751d0b65', usage=CompletionUsage(completion_tokens=503, prompt_tokens=11433, total_tokens=11936, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    # ChatCompletion(id='chatcmpl-AbqN4nKify6fxfmqjkLjKrHQ3v62r', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content='The source code related to `NDDataRef` seems to be within the `astropy/nddata/` directory. Specifically, we\'ll likely need to examine `nddata.py`, `nddata_base.py`, and possibly other related files.\n\n### Steps:\n1. **Script Creation to Reproduce the Issue:**\n   We will create a script based on the provided "How to Reproduce" section in the PR description and run it to confirm the error.\n\n2. **Inspect Relevant Source Code:**\n   We will review the source files in `astropy/nddata` to identify where the mask propagation logic is implemented.\n\n3. **Edit Code to Resolve the Issue:**\n   Modify the code to ensure that masks are appropriately propagated even when one operand lacks a mask.\n\n4. **Rerun the Reproduction Script:**\n   Verify that our changes fix the error.\n\nLet\'s create the script first to reproduce the error. We\'ll name the script `reproduce_error.py` and place it in the root directory. \n\nThe script content:\n\n```python\n# reproduce_error.py\n\nimport numpy as np\nfrom astropy.nddata import NDDataRef\n\narray = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])\nmask = np.array([[0, 1, 64], [8, 0, 1], [2, 1, 0]])\n\nnref_nomask = NDDataRef(array)\nnref_mask = NDDataRef(array, mask=mask)\n\ntry:\n    # multiply no mask by constant (no mask * no mask)\n    print(\'No mask * constant:\', nref_nomask.multiply(1., handle_mask=np.bitwise_or).mask)\n\n    # multiply no mask by itself (no mask * no mask)\n    print(\'No mask * no mask:\', nref_nomask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask)\n\n    # multiply mask by constant (mask * no mask)\n    print(\'Mask * constant:\', nref_mask.multiply(1., handle_mask=np.bitwise_or).mask)\n\nexcept Exception as e:\n    print(\'Exception:\', e)\n\ntry:\n    # multiply mask by itself (mask * mask)\n    print(\'Mask * mask:\', nref_mask.multiply(nref_mask, handle_mask=np.bitwise_or).mask)\n\n    # multiply mask by no mask (mask * no mask)\n    print(\'Mask * no mask:\', nref_mask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask)\n\nexcept Exception as e:\n    print(\'Exception:\', e)\n```\n\nI\'ll create this script and then run it to confirm the error.', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_MhPf6XuflePaFdMpA8B2Uxxt', function=Function(arguments='{"command":"create","path":"/workspace/astropy__astropy__5.2/reproduce_error.py","file_text":"import numpy as np\\nfrom astropy.nddata import NDDataRef\\n\\narray = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])\\nmask = np.array([[0, 1, 64], [8, 0, 1], [2, 1, 0]])\\n\\nnref_nomask = NDDataRef(array)\\nnref_mask = NDDataRef(array, mask=mask)\\n\\ntry:\\n    # multiply no mask by constant (no mask * no mask)\\n    print(\'No mask * constant:\', nref_nomask.multiply(1., handle_mask=np.bitwise_or).mask)\\n\\n    # multiply no mask by itself (no mask * no mask)\\n    print(\'No mask * no mask:\', nref_nomask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask)\\n\\n    # multiply mask by constant (mask * no mask)\\n    print(\'Mask * constant:\', nref_mask.multiply(1., handle_mask=np.bitwise_or).mask)\\n\\nexcept Exception as e:\\n    print(\'Exception:\', e)\\n\\ntry:\\n    # multiply mask by itself (mask * mask)\\n    print(\'Mask * mask:\', nref_mask.multiply(nref_mask, handle_mask=np.bitwise_or).mask)\\n\\n    # multiply mask by no mask (mask * no mask)\\n    print(\'Mask * no mask:\', nref_mask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask)\\n\\nexcept Exception as e:\\n    print(\'Exception:\', e)\\n"}', name='str_replace_editor'), type='function')]))], created=1733582814, model='gpt-4o-2024-05-13', object='chat.completion', service_tier=None, system_fingerprint='fp_e37d8a480e', usage=CompletionUsage(completion_tokens=939, prompt_tokens=11433, total_tokens=12372, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))

    # ChatCompletion(id='chatcmpl-AbqoH1MN4xzKY8eAhLCoTIzbqFulv', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content="The repository structure is quite extensive, but the relevant directory for our task is likely `astropy/nddata`, as it contains the `NDDataRef` class which is mentioned in the PR description.\n\n### Step 1: Create a Script to Reproduce the Error\n\nLet's create a script to reproduce the error described in the PR. This script will use the provided example to trigger the error in version 5.3.\n\n### Step 2: Execute the Script to Confirm the Error\n\nWe'll then run this script to confirm the error.\n\n### Step 3: Edit the Source Code to Resolve the Issue\n\nAfter confirming the error, we'll inspect the `NDDataRef` class and its methods to identify where the mask propagation logic is failing. We'll then make the necessary changes to ensure that the mask is correctly propagated when one of the operands does not have a mask.\n\n### Step 4: Rerun the Script to Confirm the Fix\n\nWe'll rerun the script to ensure that the error is resolved.\n\n### Step 5: Consider Edge Cases\n\nFinally, we'll think about any edge cases and ensure our fix handles them appropriately.\n\nLet's start by creating the script to reproduce the error.", refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_WpblytLpPoqKGy7hP7rLNnuy', function=Function(arguments='{"command":"create","path":"/workspace/astropy__astropy__5.2/reproduce_error.py","file_text":"import numpy as np\\nfrom astropy.nddata import NDDataRef\\n\\narray = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])\\nmask = np.array([[0, 1, 64], [8, 0, 1], [2, 1, 0]])\\n\\nnref_nomask = NDDataRef(array)\\nnref_mask = NDDataRef(array, mask=mask)\\n\\ntry:\\n    # multiply no mask by constant (no mask * no mask)\\n    print(nref_nomask.multiply(1., handle_mask=np.bitwise_or).mask)  # returns nothing, no mask,  OK\\n\\n    # multiply no mask by itself (no mask * no mask)\\n    print(nref_nomask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask)  # return nothing, no mask, OK\\n\\n    # multiply mask by constant (mask * no mask)\\n    print(nref_mask.multiply(1., handle_mask=np.bitwise_or).mask)\\nexcept TypeError as e:\\n    print(f\\"Error: {e}\\")\\n\\ntry:\\n    # multiply mask by itself (mask * mask)\\n    print(nref_mask.multiply(nref_mask, handle_mask=np.bitwise_or).mask)\\n\\n    # multiply mask by no mask (mask * no mask)\\n    print(nref_mask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask)\\nexcept TypeError as e:\\n    print(f\\"Error: {e}\\")"}', name='str_replace_editor'), type='function')]))], created=1733584501, model='gpt-4o-2024-05-13', object='chat.completion', service_tier=None, system_fingerprint='fp_e37d8a480e', usage=CompletionUsage(completion_tokens=622, prompt_tokens=11262, total_tokens=11884, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    # ChatCompletion(id='chatcmpl-AbqnHbWr7giaWrchYb6BX0UVKEQGU', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content="The repository has a complex structure with multiple directories and subdirectories. The relevant directory for our task is likely `astropy/nddata`, as it contains the `NDDataRef` class which is mentioned in the PR description.\n\n### Step 1: Create a Script to Reproduce the Error\n\nLet's create a script that reproduces the error described in the PR. This script will use the provided example to trigger the error in version 5.3.\n\n### Step 2: Execute the Script to Confirm the Error\n\nWe'll run the script to confirm that the error occurs as described.\n\n### Step 3: Edit the Source Code to Resolve the Issue\n\nAfter confirming the error, we'll inspect the `NDDataRef` class and its methods to identify where the mask propagation logic is failing. We'll then make the necessary changes to ensure that the mask is handled correctly when one of the operands does not have a mask.\n\n### Step 4: Rerun the Script to Confirm the Fix\n\nWe'll rerun the script to ensure that the error is resolved and the behavior is as expected.\n\n### Step 5: Consider Edge Cases\n\nWe'll think about any edge cases that might not be covered by the initial fix and ensure that our solution handles them as well.\n\nLet's start by creating the script to reproduce the error.", refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_nGG9FeR3jdF4E8PqRFfbR8aI', function=Function(arguments='{"command":"create","path":"/workspace/astropy__astropy__5.2/reproduce_error.py","file_text":"import numpy as np\\nfrom astropy.nddata import NDDataRef\\n\\narray = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])\\nmask = np.array([[0, 1, 64], [8, 0, 1], [2, 1, 0]])\\n\\nnref_nomask = NDDataRef(array)\\nnref_mask = NDDataRef(array, mask=mask)\\n\\n# multiply no mask by constant (no mask * no mask)\\nprint(nref_nomask.multiply(1., handle_mask=np.bitwise_or).mask)  # returns nothing, no mask,  OK\\n\\n# multiply no mask by itself (no mask * no mask)\\nprint(nref_nomask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask) # return nothing, no mask, OK\\n\\n# multiply mask by constant (mask * no mask)\\ntry:\\n    print(nref_mask.multiply(1., handle_mask=np.bitwise_or).mask)\\nexcept TypeError as e:\\n    print(e)  # Expected error\\n\\n# multiply mask by itself (mask * mask)\\nprint(nref_mask.multiply(nref_mask, handle_mask=np.bitwise_or).mask)\\n\\n# multiply mask by no mask (mask * no mask)\\ntry:\\n    print(nref_mask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask)\\nexcept TypeError as e:\\n    print(e)  # Expected error\\n"}', name='str_replace_editor'), type='function')]))], created=1733584439, model='gpt-4o-2024-05-13', object='chat.completion', service_tier=None, system_fingerprint='fp_e37d8a480e', usage=CompletionUsage(completion_tokens=633, prompt_tokens=11262, total_tokens=11895, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    # ChatCompletion(id='chatcmpl-AbqngmAXhFY9tDzxpEf6v9kPXwxPP', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content="The repository structure is quite extensive, but the relevant directory for our task is likely `astropy/nddata`, as the issue pertains to `NDDataRef` and its mask handling during arithmetic operations.\n\n### Step 1: Create a Script to Reproduce the Error\n\nLet's create a script that reproduces the error described in the PR. This script will use the provided example to trigger the `TypeError`.\n\n### Step 2: Execute the Script to Confirm the Error\n\nWe'll then run this script to confirm the error.\n\n### Step 3: Edit the Source Code to Resolve the Issue\n\nAfter confirming the error, we'll inspect the `NDDataRef` class and its arithmetic methods to identify where the mask handling logic needs to be adjusted.\n\n### Step 4: Rerun the Script to Confirm the Fix\n\nWe'll rerun the script to ensure the error is resolved.\n\n### Step 5: Consider Edge Cases\n\nFinally, we'll think about any edge cases and ensure our fix handles them appropriately.\n\nLet's start by creating the script to reproduce the error.", refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_qsK1c2aPeT2hAE0poH3OOv6C', function=Function(arguments='{"command":"create","path":"/workspace/astropy__astropy__5.2/reproduce_error.py","file_text":"import numpy as np\\nfrom astropy.nddata import NDDataRef\\n\\narray = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])\\nmask = np.array([[0, 1, 64], [8, 0, 1], [2, 1, 0]])\\n\\nnref_nomask = NDDataRef(array)\\nnref_mask = NDDataRef(array, mask=mask)\\n\\ntry:\\n    # multiply no mask by constant (no mask * no mask)\\n    print(nref_nomask.multiply(1., handle_mask=np.bitwise_or).mask)  # returns nothing, no mask,  OK\\n\\n    # multiply no mask by itself (no mask * no mask)\\n    print(nref_nomask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask)  # return nothing, no mask, OK\\n\\n    # multiply mask by constant (mask * no mask)\\n    print(nref_mask.multiply(1., handle_mask=np.bitwise_or).mask)\\nexcept TypeError as e:\\n    print(f\\"Error: {e}\\")\\n\\ntry:\\n    # multiply mask by itself (mask * mask)\\n    print(nref_mask.multiply(nref_mask, handle_mask=np.bitwise_or).mask)\\n\\n    # multiply mask by no mask (mask * no mask)\\n    print(nref_mask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask)\\nexcept TypeError as e:\\n    print(f\\"Error: {e}\\")"}', name='str_replace_editor'), type='function')]))], created=1733584464, model='gpt-4o-2024-05-13', object='chat.completion', service_tier=None, system_fingerprint='fp_e37d8a480e', usage=CompletionUsage(completion_tokens=596, prompt_tokens=11262, total_tokens=11858, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))

    # ChatCompletion(id='chatcmpl-AbqpEoJapqy9ezrfYSJDysqCkBJdL', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content="The directory structure of the repository is quite extensive. Based on the pathnames and the provided information, the most relevant directory related to the issue seems to be within `astropy/nddata`. The `NDDataRef` class is likely implemented in this directory.\n\n### Step 1: Explore the `astropy/nddata` Directory\n\nFirst, let's explore the `astropy/nddata` directory to locate the files related to the `NDDataRef` class.\n\n### Step 2: Reproduce the Error\n\nAfter identifying the relevant file and class, we'll create a script that reproduces the error. We'll execute the script to confirm the error's presence.\n\n### Step 3: Modify the Source Code\n\nWe'll then analyze the `NDDataRef` class's implementation, identify the bug, and implement the necessary changes to resolve the issue.\n\n### Step 4: Confirm the Fix\n\nAfter making the changes, we'll rerun the error reproduction script to ensure that the issue is resolved. \n\n### Step 5: Handle Edge Cases\n\nWe'll consider any edge cases during the implementation to ensure robust handling of various scenarios.\n\nLet's begin by exploring the `astropy/nddata` directory.", refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_yQAx5e3GEUIQucekXZ8rlXAt', function=Function(arguments='{"command":"ls /workspace/astropy__astropy__5.2/astropy/nddata"}', name='execute_bash'), type='function')]))], created=1733584560, model='gpt-4o-2024-05-13', object='chat.completion', service_tier=None, system_fingerprint='fp_04751d0b65', usage=CompletionUsage(completion_tokens=275, prompt_tokens=11262, total_tokens=11537, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    # ChatCompletion(id='chatcmpl-Abqp5AY7nH9R5jJnheKyYQXOJAOE6', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_D8U5eVD2bNGbO15EEIzrrODf', function=Function(arguments='{"path":"/workspace/astropy__astropy__5.2/reproduce_error.py","content":"import numpy as np\\nfrom astropy.nddata import NDDataRef\\n\\ndef main():\\n    array = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])\\n    mask = np.array([[0, 1, 64], [8, 0, 1], [2, 1, 0]])\\n\\n    nref_nomask = NDDataRef(array)\\n    nref_mask = NDDataRef(array, mask=mask)\\n\\n    try:\\n        result = nref_mask.multiply(1., handle_mask=np.bitwise_or).mask\\n        print(\'mask * no mask (constant):\', result)\\n    except Exception as e:\\n        print(\'mask * no mask (constant) Exception:\', e)\\n\\n    try:\\n        result = nref_mask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask\\n        print(\'mask * no mask:\', result)\\n    except Exception as e:\\n        print(\'mask * no mask Exception:\', e)\\n\\nif __name__ == \'__main__\':\\n    main()\\n"}', name='create_file'), type='function')]))], created=1733584551, model='gpt-4o-2024-05-13', object='chat.completion', service_tier=None, system_fingerprint='fp_04751d0b65', usage=CompletionUsage(completion_tokens=281, prompt_tokens=11262, total_tokens=11543, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    # ChatCompletion(id='chatcmpl-Abqon76VlAxMT6gK2VXDmOIdg14Tk', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content="The relevant directories and files for NDDataRef are likely within the `astropy/nddata` directory. To start, we should focus on `astropy/nddata` and find the implementation of NDDataRef, and understand how mask handling is being performed.\n\nLet's list the contents of `astropy/nddata` to narrow down our search.", refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_QTrjHWJPXWvBMxmaucZNnrjP', function=Function(arguments='{"command":"ls -R /workspace/astropy__astropy__5.2/astropy/nddata"}', name='execute_bash'), type='function')]))], created=1733584533, model='gpt-4o-2024-05-13', object='chat.completion', service_tier=None, system_fingerprint='fp_84ac7e2412', usage=CompletionUsage(completion_tokens=108, prompt_tokens=11262, total_tokens=11370, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    # ChatCompletion(id='chatcmpl-Abqu52caVYIq8E3vbZQxJRU2gBVte', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_zCjhdgKWxUi0n4KMVrZPGVbt', function=Function(arguments='{"path":"/workspace/astropy__astropy__5.2/reproduce_error.py","file_text":"import numpy as np\\nfrom astropy.nddata import NDDataRef\\n\\narray = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])\\nmask = np.array([[0, 1, 64], [8, 0, 1], [2, 1, 0]])\\n\\nnref_nomask = NDDataRef(array)\\nnref_mask = NDDataRef(array, mask=mask)\\n\\ntry:\\n    print(nref_nomask.multiply(1., handle_mask=np.bitwise_or).mask)\\n    print(\'No error with no mask * no mask (constant)\')\\nexcept TypeError as e:\\n    print(f\'Error with no mask * no mask (constant): {e}\')\\n\\ntry:\\n    print(nref_nomask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask)\\n    print(\'No error with no mask * no mask\')\\nexcept TypeError as e:\\n    print(f\'Error with no mask * no mask: {e}\')\\n\\ntry:\\n    print(nref_mask.multiply(1., handle_mask=np.bitwise_or).mask)\\n    print(\'No error with mask * no mask (constant)\')\\nexcept TypeError as e:\\n    print(f\'Error with mask * no mask (constant): {e}\')\\n\\ntry:\\n    print(nref_mask.multiply(nref_mask, handle_mask=np.bitwise_or).mask)\\n    print(\'No error with mask * mask\')\\nexcept TypeError as e:\\n    print(f\'Error with mask * mask: {e}\')\\n\\ntry:\\n    print(nref_mask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask)\\n    print(\'No error with mask * no mask\')\\nexcept TypeError as e:\\n    print(f\'Error with mask * no mask: {e}\')\\n"}', name='create_file'), type='function')]))], created=1733584861, model='gpt-4o-2024-05-13', object='chat.completion', service_tier=None, system_fingerprint='fp_84ac7e2412', usage=CompletionUsage(completion_tokens=454, prompt_tokens=11262, total_tokens=11716, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    # ChatCompletion(id='chatcmpl-Abqud6GqQvzxsPu4gj6Ez9sRw4PDf', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content="The error described in the PR relates to the `NDDataRef` class in the `astropy.nddata` package. Within version 5.2, there were no problems when handling masks in operations even when one operand did not have a mask. This changed in version 5.3, leading to a `TypeError` during mask operations.\n\nTo resolve this issue, I'll do the following:\n\n1. **Create a script to reproduce the error in v5.2.** This script will confirm that there is no error in v5.2.\n2. **Review the code in v5.2 for the `NDDataRef` arithmetic operations** to understand why it is working without error and compare it with v5.3.\n3. **Identify the changes in v5.3 that introduced the error** and correct them to ensure the mask operations function as expected.\n4. **Rerun the script to confirm the error is resolved.**\n\nTo start, I'll create a reproduction script. Let's first confirm the location of `NDDataRef` in the `astropy.nddata` module and proceed accordingly.", refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_o5mqt16g2zQg3WVTziMlJPF8', function=Function(arguments='{"command":"ls -R /workspace/astropy__astropy__5.2/astropy/nddata"}', name='execute_bash'), type='function')]))], created=1733584895, model='gpt-4o-2024-05-13', object='chat.completion', service_tier=None, system_fingerprint='fp_04751d0b65', usage=CompletionUsage(completion_tokens=264, prompt_tokens=11262, total_tokens=11526, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    # openai: 726.0 [947, 592, 619, 785, 604, 598, 895, 611, 596, 913, 897, 474, 605, 950, 947, 789, 618, 923, 572, 586]
    # aigcbe: 374.2 [340, 204, 710, 372, 36,  749, 754, 413, 103, 877, 373, 246, 453, 131, 391, 411, 190, 196, 275, 260]
    # mini 359.95 [36, 534, 505, 519, 298, 273, 36, 534, 298, 533, 278, 511, 641, 525, 301, 532, 36, 556, 36, 36]
    # mini 328.55 [262, 511, 311, 558, 298, 277, 679, 829, 296, 703, 311, 446, 258, 423, 354, 595, 289, 301, 277, 281]
    # aigcbe: 429.9 [476, 629, 335, 450, 100, 277, 227, 277, 937, 780, 112, 739, 115, 923, 116, 495, 512, 91, 175, 833]
    # openai: 490.0 [916, 624, 621, 98, 278, 876, 257, 478, 306, 338, 633, 593, 833, 480, 467, 264, 259, 281, 951, 247]


def test_message_post_processing():
    from litellm import Message

    def _post_process_messages(
        messages: list[Message], max_message_iterations: int = -1
    ) -> list[Message]:
        if max_message_iterations < 0:
            return messages
        else:
            post_processed_messages = []
            if len(messages) > 0 and messages[0].role == 'system':
                post_processed_messages.append(messages[0])
            else:
                raise ValueError('The first message must be a system message')
            if len(messages) > 1 and messages[1].role == 'user':
                post_processed_messages.append(messages[1])
            else:
                raise ValueError('The second message must be a user message')
            if len(messages) > 2 and messages[2].role == 'assistant':
                post_processed_messages.append(messages[2])
            else:
                raise ValueError('The third message must be an assistant message')
            if len(messages) > 3 and messages[3].role == 'tool':
                post_processed_messages.append(messages[3])

            if len(messages) > len(post_processed_messages) + max_message_iterations:
                post_processed_messages.append(
                    Message(role='user', content="You are doing great! Let's continue.")
                )
                while messages[-max_message_iterations].role != 'assistant':
                    max_message_iterations -= 1
                post_processed_messages.extend(messages[-max_message_iterations:])
            else:
                post_processed_messages.extend(messages[len(post_processed_messages) :])

        return post_processed_messages

    data = json.load(
        open(
            '/hdd2/zzr/OpenHands-fn-calling/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-4o-mini-2024-07-18_maxiter_100_N_v2.1-no-hint-run_1_full/llm_completions/django__django-10880/gpt-4o-mini-2024-07-18-1733597466.6885712.json'
        )
    )

    messages = [
        Message(
            role=msg['role'],
            content=msg['content'][0]['text'] if len(msg['content']) > 0 else '',
            function_call=msg['function_call'] if 'function_call' in msg else None,
            tool_calls=msg['tool_calls'] if 'tool_calls' in msg else None,
        )
        for msg in data['messages']
    ]
    # import pdb; pdb.set_trace()
    post_processed_messages = _post_process_messages(messages, max_message_iterations=5)
    print(post_processed_messages)


def test_clustering():
    import numpy as np
    from sklearn.cluster import DBSCAN, OPTICS, KMeans
    from sklearn.metrics import silhouette_score

    def select_by_kmeans(embeddings: list[float]) -> int:
        # 自动选择最优簇数并进行聚类
        # 将嵌入转换为数组
        embedding_array = np.array(embeddings)

        # 初始化变量
        max_k = min(len(embedding_array) - 1, 10)  # 最大簇数
        best_k = 2
        best_score = -1

        # 计算每个簇数的轮廓系数
        for k in range(2, max_k):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embedding_array)
            score = (
                silhouette_score(embedding_array, labels)
                if len(np.unique(labels)) > 1
                else 0
            )

            if score > best_score:
                best_score = score
                best_k = k

        # 使用最佳簇数重新进行K-means聚类
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        labels = kmeans.fit_predict(embedding_array)

        # 找到最大簇
        unique, counts = np.unique(labels, return_counts=True)
        max_cluster_label = unique[np.argmax(counts)]

        # 找到最大簇的类中心
        max_cluster_center = kmeans.cluster_centers_[max_cluster_label]

        # 计算最大簇中每个点到类中心的距离
        max_cluster_indices = [
            i for i, label in enumerate(labels) if label == max_cluster_label
        ]
        distances = [
            np.linalg.norm(np.array(embeddings[i]) - max_cluster_center)
            for i in max_cluster_indices
        ]

        # 找到距离类中心最近的选项
        closest_index = max_cluster_indices[np.argmin(distances)]

        return closest_index

    def select_by_dbscan(embeddings: list[float]) -> int:
        # 将嵌入转换为数组
        embedding_array = np.array(embeddings)

        # 使用DBSCAN进行聚类
        dbscan = DBSCAN(eps=0.5, min_samples=2)  # 可以根据数据调整eps和min_samples
        labels = dbscan.fit_predict(embedding_array)

        # 找到最大簇
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        max_cluster_label = unique[np.argmax(counts)]

        # 找到最大簇的类中心
        max_cluster_indices = [
            i for i, label in enumerate(labels) if label == max_cluster_label
        ]
        max_cluster_center = np.mean(
            [embeddings[i] for i in max_cluster_indices], axis=0
        )

        # 计算最大簇中每个点到类中心的距离
        distances = [
            np.linalg.norm(np.array(embeddings[i]) - max_cluster_center)
            for i in max_cluster_indices
        ]

        # 找到距离类中心最近的选项
        closest_index = max_cluster_indices[np.argmin(distances)]
        return closest_index

    def select_by_optics(embeddings: list[float]) -> int:
        # 使用OPTICS进行聚类
        # 将嵌入转换为数组
        embedding_array = np.array(embeddings)

        # 使用OPTICS进行聚类
        optics = OPTICS(min_samples=2)  # 可以根据数据调整min_samples
        labels = optics.fit_predict(embedding_array)
        print(labels)
        # 找到最大簇
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        max_cluster_label = unique[np.argmax(counts)]

        # 找到最大簇的类中心
        max_cluster_indices = [
            i for i, label in enumerate(labels) if label == max_cluster_label
        ]
        max_cluster_center = np.mean(
            [embeddings[i] for i in max_cluster_indices], axis=0
        )

        # 计算最大簇中每个点到类中心的距离
        distances = [
            np.linalg.norm(np.array(embeddings[i]) - max_cluster_center)
            for i in max_cluster_indices
        ]

        # 找到距离类中心最近的选项
        closest_index = max_cluster_indices[np.argmin(distances)]

        return closest_index

    def test_select_by_kmeans():
        embeddings = [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ]
        result = select_by_kmeans(embeddings)
        print(f'select_by_kmeans result: {result}')

    def test_select_by_dbscan():
        embeddings = [
            [1.0, 2.0],
            [1.1, 2.1],
            [1.1, 2.1],
            [1.2, 2.2],
            [1.2, 2.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [50.0, 50.0],
            [50.1, 50.1],
            [50.2, 50.2],
        ]
        result = select_by_dbscan(embeddings)
        print(f'select_by_dbscan result: {result}')

    def test_select_by_optics():
        embeddings = [
            [1.0, 2.0],
            [1.1, 2.1],
            [1.1, 2.1],
            [1.2, 2.2],
            [1.2, 2.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [50.0, 50.0],
            [50.1, 50.1],
            [50.2, 50.2],
        ]
        result = select_by_optics(embeddings)
        print(f'select_by_optics result: {result}')

    # 运行测试用例
    test_select_by_kmeans()
    test_select_by_dbscan()
    test_select_by_optics()


def test_parameter_parsing():
    def get_all_keys(d):
        """递归获取字典中的所有键值"""
        keys = []
        for key, value in d.items():
            keys.append(key)
            if isinstance(value, dict):
                keys.extend(get_all_keys(value))
        return keys

    def get_all_key_values(d):
        """递归获取字典中的所有键值对

        Args:
            d: 要处理的字典

        Returns:
            list: 包含所有键值对的列表，格式为 [(key_path, value), ...]
        """
        items = []
        for key, value in d.items():
            if isinstance(value, dict):
                items.extend(get_all_key_values(value))
            items.append((key, value))
        return items

    raw_arguments = {
        'fields': {
            'key': 'command',
            'path': {
                'string_value': '/workspace/astropy__astropy__5.2/astropy/nddata/nddataref.py'
            },
            'value': {'string_value': 'view'},
        }
    }
    arguments = {}
    all_keys = get_all_keys(raw_arguments)
    if 'fields' in all_keys:
        all_key_values = get_all_key_values(raw_arguments)
        _arguments = {k: v for k, v in all_key_values}
        for key in _arguments:
            if isinstance(_arguments[key], dict):
                for v_key in _arguments[key]:
                    if 'value' in v_key:
                        break
                _arguments[key] = _arguments[key][v_key]

        if 'key' in _arguments and 'value' in _arguments:
            _arguments[_arguments['key']] = _arguments['value']
        for key in _arguments:
            if 'key' in key or 'value' in key or 'fields' in key:
                continue
            arguments[key] = _arguments[key]
    else:
        arguments = raw_arguments
    print(f'New arguments: {arguments}')


if __name__ == '__main__':
    test_gemini_api()
    # test_litellm()
    # test_openrouter()
    # test_gemini()
    # test_openai()
    # test_anthropic()
    # test_anthropic_2()
    # test_openai2()
    # test_message_post_processing()
    # test_clustering()
    # test_parameter_parsing()
