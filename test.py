import requests
import json

response = requests.post(
    'http://localhost:8000/query/stream',
    json={'query': 'Tell me about AI'},
    stream=True
)

# Add error handling and debug prints
if response.status_code != 200:
    print(f"Error: Server returned status code {response.status_code}")
    print(f"Response text: {response.text}")
    exit(1)

print("Connected to server, waiting for response...")

for line in response.iter_lines():
    if not line:
        continue
        
    try:
        # Remove "data: " prefix and parse JSON
        line_text = line.decode('utf-8')
        if not line_text.startswith('data: '):
            print(f"Unexpected line format: {line_text}")
            continue
            
        data = json.loads(line_text.replace('data: ', ''))
        
        if 'response' in data:
            print(data['response'], end='', flush=True)
        elif 'sources' in data:
            print('\nSources:', data['sources'])
        else:
            print(f"Unexpected data format: {data}")
            
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw line: {line}")
    except Exception as e:
        print(f"Error processing line: {e}")