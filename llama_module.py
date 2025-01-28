from ollama import chat

def LlamaHandler(prompt):
    stream = chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

if __name__ == "__main__":
    a = LlamaHandler('cat')
