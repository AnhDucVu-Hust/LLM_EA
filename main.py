import json
import os
import numpy as np
from gpt_api import call_gpt
import openai
import tqdm
import re
import argparse
import string
import random
from templates.context_gen_template import template_1
api_key="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"
openai.api_key=api_key

def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = template_1
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx+1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt

def sample_context(contexts,n):
    return random.sample(contexts,min(n,len(contexts)))

def post_process_response(response):
    if len(response['choices'])==0 or response is None or response['choices'][0].finish_reason == "length":
        return []
    raw_instructions = re.split(r"\n\d+\s?\. ", response["choices"][0].text)
    instructions = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if inst == "":
            continue
        if inst[0] in string.punctuation:
            continue
        if not inst[0].isascii():
            continue
        instructions.append(inst)
    return instructions

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_contexts",
        type=int,
        required=True,
        default=1000
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci-002",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
    )
    args = parser.parse_args()
    seed_contexts = [json.loads(l) for l in open("./data/seed_contexts.jsonl","r",encoding="UTF-8")]
    seed_instruction = [t["instruction"] for t in seed_contexts]
    output_dir = "./data/generated_contexts/"
    os.makedirs(output_dir,exist_ok=True)
    machine_contexts = []
    if os.path.exists(os.path.join(output_dir,"generated_contexts_gpt.jsonl")):
        with open(os.path.join(output_dir, "generated_contexts_gpt.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_contexts.append(instruction_info["instruction"])
        print(f"Loaded {len(machine_contexts)} machine-generated instructions")
    progress_bar = tqdm.tqdm(total=args.num_contexts)
    if machine_contexts:
        progress_bar.update(len(machine_contexts))
    with open(os.path.join(output_dir, "generated_contexts_gpt.jsonl"), "a",encoding="UTF-8") as fout:
        while len(machine_contexts) < args.num_contexts:
            batch_inputs = []
            for _ in range(args.request_batch_size):
                prompt_instructions = sample_context(
                    machine_contexts,
                    n=2)
                prompt_instructions += random.sample(seed_instruction,3)
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(prompt_instructions)
                batch_inputs.append({prompt})
            results = call_gpt(
                engine=args.engine,
                prompts = batch_inputs,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences="\n\n",
                logprobs=False,
                n=1,
            )
            instructions = []
            all_metadata = []
            for result in results:
                new_instructions = post_process_response(result["response"])
                instructions += new_instructions

            for inst in instructions:
                all_instructions = seed_instruction + machine_contexts
                machine_contexts.append(inst)
                fout.write(json.dumps({
                    "instruction": inst,
                }) + "\n")
                progress_bar.update(1)





