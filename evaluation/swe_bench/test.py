import json

from openai import OpenAI
from tqdm import tqdm


def test_openai():
    api_key = 'sk-isa0ZCsGSRizprusXPMdwIj1LstdPxYtjmkjMJHAtIqRL7zi'
    # api_key = 'sk-M3bKoucx7o9TSEkK0eC5249f951644E69d880eC269Df061d'

    client = OpenAI(api_key=api_key, base_url='https://api2.aigcbest.top/v1')

    # 定义函数
    def get_weather(location):
        return {'Beijing': '晴天'}

    # 定义函数调用
    response = client.chat.completions.create(
        # model="gpt-4o-2024-05-13",
        # model="gpt-3.5-turbo",
        model='claude-3-5-sonnet-20241022',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': "What's the weather like in Beijing?"},
        ],
        n=4,
        functions=[
            {
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
            }
        ],
        function_call={'name': 'get_weather', 'arguments': {'location': 'Beijing'}},
    )

    # 打印函数调用结果
    print(response.choices)


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
    # api_key = 'sk-M3bKoucx7o9TSEkK0eC5249f951644E69d880eC269Df061d'
    # client = OpenAI(api_key=api_key, base_url='https://api2.aigcbest.top/v1')

    # api_key = 'hk-4qt48b100004811414c08bcb08e49f7a42b44edde99c63e1'
    # client = OpenAI(api_key=api_key, base_url='https://api.openai-hk.com/v1')

    api_key = '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'
    client = OpenAI(api_key=api_key)

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
            model='gpt-4o-mini',
            messages=msgs,
            tools=data['kwargs']['tools'],
            temperature=0.2,
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
    # openai: [947, 592, 619, 785, 604, 598, 895, 611, 596, 913, 897, 474, 605, 950, 947, 789, 618, 923, 572, 586]
    # aigcbe: [340, 204, 710, 372, 36,  749, 754, 413, 103, 877, 373, 246, 453, 131, 391, 411, 190, 196, 275, 260]
    # mini 359.95 [36, 534, 505, 519, 298, 273, 36, 534, 298, 533, 278, 511, 641, 525, 301, 532, 36, 556, 36, 36]
    # mini 328.55 [262, 511, 311, 558, 298, 277, 679, 829, 296, 703, 311, 446, 258, 423, 354, 595, 289, 301, 277, 281]


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


if __name__ == '__main__':
    test_openai()
    # test_anthropic()
    # test_anthropic_2()
    # test_openai2()
    # test_message_post_processing()
