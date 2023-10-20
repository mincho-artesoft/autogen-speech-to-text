import json
import openai


openai.api_key = 'sk-'
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "make me a burger recipe and give it to me in json format"}]
)

print(response)

response_text = response['choices'][0]['message']['content']

recipe_json = json.loads(response_text)

print(recipe_json)
