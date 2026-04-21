from models import *
from utils import *
from icecream import ic
import json, yaml

SYS_MSG = f"""
EXTRACT {','.join(NewsSummaryBase.model_fields.keys())} FROM INPUT HAVING value NOT IN (NULL, '', [])
RESPONSE FORMAT: JSON
FIELDS:
{NewsSummaryBase.schema()}
"""
INST_MSG = f"EXTRACT * FROM INPUT:\n{input}"

create_msg = lambda text: [
    { "role": "system", "content": SYS_MSG },
    { "role": "user", "content": INST_MSG.format(text) }
]

def load_inputs():
    return [create_msg(item['text']) for item in load_data_from_directory("data/digests/*.json")[:128]]

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main(model_name):    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    create_text = lambda messages: tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    texts = list(map(create_text, load_inputs()))
    llm = LLM(model_name)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.00, top_k=20, min_p=0.0, presence_penalty=2.0, repetition_penalty=1.0, max_tokens=16384)
    outputs = llm.generate(texts, sampling_params=sampling_params)
    data = [t.outputs[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```") for t in outputs]
    os.makedirs(".test", exist_ok=True)
    with open(".test/outputs.json", "w") as f:
        f.write("[\n")
        f.write(",\n".join(data))
        f.write("\n]")
        # json.dump(data, f, indent=2)

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
if __name__=="__main__":
    main(MODEL_NAME)

