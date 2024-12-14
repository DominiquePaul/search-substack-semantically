import requests
import json

# First question
response1 = requests.post(
    'http://localhost:8000/query',
    json={
        'messages': [
            {
                'role': 'system',
                'content': 'You are a helpful assistant that can recommend articles on a given topic. You can recommend up to 5 articles. Do not recommend the same url twice. Always provide title, substack, short description of the post and url.'
            },
            {
                'role': 'user',
                'content': 'Tell me about AI'
            }
        ],
        'stream': True
    },
    stream=True
)

def process_stream(response):
    if response.status_code != 200:
        print(f"Error: Server returned status code {response.status_code}")
        print(f"Response text: {response.text}")
        return None

    print("\nWaiting for response...\n")
    assistant_response = ""

    for line in response.iter_lines():
        if not line:
            continue
            
        try:
            line_text = line.decode('utf-8')
            if line_text == "data: [DONE]":
                break
                
            if not line_text.startswith('data: '):
                print(f"Unexpected line format: {line_text}")
                continue
                
            data = json.loads(line_text.replace('data: ', ''))
            
            if 'choices' in data:
                chunk = data['choices'][0]['delta'].get('content', '')
                assistant_response += chunk
                print(chunk, end='', flush=True)
            elif 'sources' in data:
                print('\nSources:', data['sources'])
            else:
                print(f"Unexpected data format: {data}")
                
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw line: {line}")
        except Exception as e:
            print(f"Error processing line: {e}")
    
    return assistant_response

# Process first response
assistant_response1 = process_stream(response1)

# Second question (including previous conversation)
response2 = requests.post(
    'http://localhost:8000/query',
    json={
        'messages': [
            {
                'role': 'system',
                'content': 'You are a helpful assistant that can recommend articles on a given topic. You can recommend up to 5 articles. Do not recommend the same url twice. Always provide title, substack, short description of the post and url.'
            },
            {
                'role': 'user',
                'content': 'Tell me about AI'
            },
            {
                'role': 'assistant',
                'content': assistant_response1
            },
            {
                'role': 'user',
                'content': 'What did I ask you about initially?'
            }
        ],
        'stream': True
    },
    stream=True
)

# Process second response
process_stream(response2)