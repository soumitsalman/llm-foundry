from models import *
from utils import *
from icecream import ic
import json, yaml, random

# TOOL CALL FOR INSTRUCTS
# SYS_MSG = """Tools: [
#     {{
#         "name": "extract_structured_digest",
#         "description": "Extracts structured digest fields and summary points from given input.",
#         "parameters": {schema}
#     }} 
# ]
# Avoid: markdown, prose, code fences, empty values, null values, assumptions, implied information
# Response: Tool-call ONLY
# """
# INST_MSG = "EXTRACT {fields} FROM:\n{text}"

# PROMPT FOR EXTRACTION
SYS_MSG = """
RESPONSE FORMAT: JSON
SCHEMA: {schema}
AVOID: markdown, prose, code fences, empty values, null values, assumptions, implied information
"""
INST_MSG = "EXTRACT {fields} FROM:\n{text}"

create_msg = lambda text: [
    { "role": "system", "content": SYS_MSG.format(schema=NewsSummaryBase.model_json_schema()) },
    { "role": "user", "content": INST_MSG.format(fields=",".join(NewsSummaryBase.model_fields.keys()), text=text) }
]

def load_inputs():
    return [create_msg(item['text']) for item in random.sample(load_data_from_directory("data/digests/*.json"), 32)]

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main(model_name):    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    create_text = lambda messages: tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    texts = list(map(create_text, load_inputs()))
    llm = LLM(model_name)
    sampling_params = SamplingParams(temperature=0.15, top_p=0.95, top_k=40, repetition_penalty=1.05, max_tokens=32768)
    outputs = llm.generate(texts, sampling_params=sampling_params)
    data = [json.dump(t.outputs[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")) for t in outputs]
    os.makedirs(".test", exist_ok=True)
    with open(".test/outputs.json", "w") as f:
        # f.write("[\n")
        # f.write(",\n".join(data))
        # f.write("\n]")
        json.dump(data, f, indent=2)

# MODEL_NAME = "Qwen/Qwen3.5-0.8B" --> not so good
# MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct" --> great!
MODEL_NAME = "LiquidAI/LFM2-1.2B-Extract"
if __name__=="__main__":
    main(MODEL_NAME)

