import openai
import time
from datetime import datetime
def call_gpt(engine,prompts,max_tokens,temperature,top_p,frequency_penalty,
             presence_penalty,stop_sequences,logprobs,n,retries=3,api_key=None,organization=None):
    target_length = max_tokens
    response = None
    if api_key is not None:
        openai.api_key=api_key
    if organization is not None:
        openai.organization=organization
    retry_count=0
    backoff_time=30
    while retry_count <= retries:
        try:
            response = openai.completions.create(
                model = engine,
                prompt = prompts,
                max_tokens = target_length,
                temperature= temperature,
                top_p = top_p,
                frequency_penalty = frequency_penalty,
                presence_penalty = presence_penalty,
                stop = stop_sequences,
                logprobs = 1,
                n = n,
            )
            break
        except openai.OpenAIError as e:
            print(f"Error: {e}.")
            if "Please reduce your prompt" in str(e):
                target_length = int(target_length * 0.8)
                print(f"Reducing target length to {target_length}, retrying...")
            else:
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
            retry_count +=1
    if isinstance(prompts, list):
        results = []
        for j, prompt in enumerate(prompts):
            data = {
                "prompt": prompt,
                "response": {"choices": response.choices[j * n: (j + 1) * n]} if response else None,
                "created_at": str(datetime.now()),
            }
            results.append(data)
        return results
    else:
        data = {
            "prompt": prompts,
            "response": response,
            "created_at": str(datetime.now()),
        }
        return [data]