import requests
import json


class LlamaHandler():
    recorded_text = ''
    url = 'http://localhost:11434/api/chat'
    def llama3(self, prompt):
        data = {
            "model": "llama3.2",
            "messages": [
                {
                    "role" : "user",
                    "content" : prompt

                }
            ],
            "stream":False

        }
        headers ={
                "Content-Type" :'application/json'

        }

        response = requests.post(self.url,headers=headers, json = data)
        print(response)
        return(response.json()['message']['content'])


    def specific_prompt(self, n):
        prompt = "The following is the sentence/ main keywords being spoken by a person," + "please edit (rephrase) it such that it makes more sense;" + "generate" + str(n) + "sentences as the output, remember to not give any other text in the output." + "(Sample: guide please -- Please guide me.):"
        return prompt

    def answer(self, keywords):
        response = self.llama3(self.specific_prompt(3) + str(keywords))
        return response

