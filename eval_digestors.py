from models import *
from utils import *
from icecream import ic
import json, yaml, random

# TOOL CALL FOR INSTRUCTS
# Avoid: markdown, prose, code fences, empty values, null values, assumptions, implied information
# Response Format: JSON FUNCTION CALL ONLY

TOOL_SCHEMA = [
    {
        "name": "extract_fields",
        "description": "Extracts specified fields and contents from input.",
        "parameters": NewsSummaryBase.model_json_schema()
    }
]
SYS_MSG = f"List of tools: {json.dumps(TOOL_SCHEMA)}"
INST_MSG = "EXTRACT FIELDS {fields} FROM\n```{kind}\n{text}\n```"

# PROMPT FOR EXTRACTION
# SYS_MSG = """
# RESPONSE FORMAT: JSON
# SCHEMA: {schema}
# AVOID: markdown, prose, code fences, empty values, null values, assumptions, implied information
# """
# INST_MSG = "EXTRACT {fields} FROM:\n{text}"

create_msg = lambda text: [
    { "role": "system", "content": SYS_MSG },
    { "role": "user", "content": INST_MSG.format(fields=",".join(NewsSummaryBase.model_fields.keys()), kind="blog", text=text[:32768]) }
]

def load_inputs(count=32):
    return [create_msg(item['text']) for item in random.sample(load_data_from_directory("data/digests/*.json"), count)]


def main_transformers(model_name, sampling_params, messages):
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype="float16")
    
    input_ids = tokenizer.apply_chat_template(
        messages, 
        truncation=True, 
        padding_side='left',
        padding=True, 
        return_tensors="pt", 
        return_dict=True, 
        tokenize=True, 
        add_generation_prompt=True
    ).to(model.device)
    input_lengths = [len(ids) for ids in input_ids.input_ids]
    output_ids = model.generate(**input_ids, do_sample=True, **sampling_params)
    
    return tokenizer.batch_decode([o[i_len:] for o, i_len in zip(output_ids, input_lengths)], skip_special_tokens=True)
    # data = [ic(t.strip().removeprefix("```json").removeprefix("```").removesuffix("```")) for t in outputs]

def main_vllm(model_name, sampling_params, messages):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    llm = LLM(model_name)
    outputs = llm.generate(inputs, sampling_params=SamplingParams(**sampling_params))

    return [ic(t.outputs[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")) for t in outputs]
    
def save_outputs(outputs):   
    os.makedirs(".test", exist_ok=True)
    with open(".test/outputs.json", "w") as f:
        f.write("[\n")
        f.write(",\n".join(outputs))
        f.write("\n]")
        # json.dump(data, f, indent=2)


# MODEL_NAME = "Qwen/Qwen3.5-0.8B" --> not so good
# MODEL_NAME = "Qwen/Qwen3.6-35B-A3B" 
# sampling_params = SamplingParams(temperature=1.0, top_p=1.0, top_k=40, min_p=0.0, presence_penalty=2.0, repetition_penalty=1.0, max_tokens=32768)
model_name = "LiquidAI/LFM2.5-1.2B-Instruct" # --> great!
sampling_params = {"temperature": 0.3, "top_k": 30, "repetition_penalty": 1.0, "max_tokens": 32768}
# MODEL_NAME = "LiquidAI/LFM2-1.2B-Extract" --> gets stuck with vllm
# MODEL_NAME = "LiquidAI/LFM2-350M-Extract" --> gets stuck with vllm, but works with transformers generate
if __name__=="__main__":
    messages = load_inputs(count=64)
    res = main_vllm(model_name, sampling_params, messages)
    # res = main_transformers(model_name, sampling_params, messages)
    save_outputs(res)

