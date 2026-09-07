import os
import requests

response = requests.post(
    'https://api.blackbox.ai/chat/completions',
    headers={
        'Authorization': f'Bearer {os.environ['BLACKBOX_API_KEY']}',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'inclusionai/ling-3.0-flash-fin-free',
        'messages': [{'role': 'user', 'content': 'Hello, Blackbox!'}],
        'stream': True,
    },
)
data = response.json()