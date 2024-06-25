import json
import requests

model = 'tinyllama'
prompt = 'Who is the president of the USA?'

def generate(prompt):
    r = requests.post('http://localhost:11434/api/generate',
                      json={
                          'model': model,
                          'prompt': prompt,
                      },
                      stream=True)
    r.raise_for_status()

    response_parts = []
    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')
        response_parts.append(response_part)

        if 'error' in body:
            raise Exception(body['error'])

        if body.get('done', False):
            return ''.join(response_parts)

if __name__ == "__main__":
    response = generate(prompt)
    print(f"Response from {model}: {response}")
