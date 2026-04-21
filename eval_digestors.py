from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from models import *
from utils import *

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

SYS_MSG = f"""
EXTRACT {','.join(NewsSummaryBase.model_fields.keys())} FROM INPUT INTO raw_response
{NewsSummaryBase.schema()}
EXCLUDE field FROM raw_response WHERE value IS NULL OR value IS EMPTY INTO response
RESPONSE FORMAT: JSON
"""
INST_MSG = f"INPUT:\n{input}"

create_msg = lambda text: [
    { "role": "system", "content": SYS_MSG },
    { "role": "user", "content": INST_MSG.format(text) }
]


def load_inputs():
    return [create_msg(item['text']) for item in load_data_from_directory("data/digests/*.json")[:128]]

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    create_text = lambda messages: tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    texts = list(map(create_text, load_inputs()))
    llm = LLM(MODEL_NAME)
    sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=512)
    outputs = llm.generate(texts, sampling_params=sampling_params)
    data = [json.loads(t.strip().removeprefix("```json").removeprefix("```").removesuffix("```")) for t in outputs]
    with open(".test/outputs.json", "w") as f:
        json.dump(data, f, indent=2)

if __name__=="__main__":
    main()

