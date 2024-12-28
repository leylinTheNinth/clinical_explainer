import os
from groq import Groq

def print_available_nlexp_models(api_key):
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    response = client.models.list()
    available_models = []
    for model_available in response["data"]:
        # print("Model: ", model_available["id"])
        try:
            completion = client.chat.completions.create(
                model=model_available["id"],
                messages=[
                    {
                        "role": "user",
                        "content": "Hi"
                    }
                ],
                temperature=1,
                max_tokens=150,
                top_p=1,
                stream=False,
                stop=None,
            )
            available_models.append(model_available["id"])
            del completion
        except Exception as e:
            # print(f"Error with model {model_available['id']}: {e}")
            pass
    return available_models