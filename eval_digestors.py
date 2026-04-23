from models import *
from utils import *
from icecream import ic
import json, yaml, random
from typing import Any, Optional, Type


def _strip_json_fences(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _safe_model_dump(item: Optional[Digest]):
    if item is None:
        return None
    return item.model_dump(mode="json", exclude_none=True, exclude_unset=True, exclude_defaults=True)

# TOOL CALL FOR INSTRUCTS
# Avoid: markdown, prose, code fences, empty values, null values, assumptions, implied information
# Response Format: JSON FUNCTION CALL ONLY

TOOL_NAME = "extract_fields"
DEFAULT_MAX_TEXT_CHARS = 32768


def build_tool_schema(model_type: Type[BaseModel]):
    return {
        "name": TOOL_NAME,
        "description": "Extracts specified fields and contents from input.",
        "parameters": model_type.model_json_schema(),
    }


def build_chat_tools(model_type: Type[BaseModel]):
    return [
        {
            "type": "function",
            "function": build_tool_schema(model_type),
        }
    ]


def build_system_message(model_type: Type[BaseModel]) -> str:
    return f"List of tools: {json.dumps([build_tool_schema(model_type)])}"


def build_user_message(model_type: Type[BaseModel], text: str, kind: str = "blog") -> str:
    return f"EXTRACT FIELDS {','.join(model_type.model_fields.keys())} FROM\n```{kind}\n{text[:DEFAULT_MAX_TEXT_CHARS]}\n```"


STRUCTURED_OUTPUT_SYS_MSG = """
RETURN=JSON object matching schema
EXCLUDE=unspecified data, implied assessments, assumptions
REMOVE=empty or null fields
AVOID=markdown, prose, code fences, null placeholders, implied information, assumptions
"""
STRUCTURED_OUTPUT_INST_MSG = """
EXTRACT {fields} FROM content IF specified
=== content ===
{text}
"""

# PROMPT FOR EXTRACTION
# SYS_MSG = """
# RESPONSE FORMAT: JSON
# SCHEMA: {schema}
# AVOID: markdown, prose, code fences, empty values, null values, assumptions, implied information
# """
# INST_MSG = "EXTRACT {fields} FROM:\n{text}"

def create_msg(text):
    return [
        {"role": "system", "content": build_system_message(Digest)},
        {"role": "user", "content": build_user_message(Digest, text)},
    ]


def create_structured_output_msg(text):
    return [
        {"role": "system", "content": STRUCTURED_OUTPUT_SYS_MSG.format(schema=Digest.model_json_schema())},
        {
            "role": "user",
            "content": STRUCTURED_OUTPUT_INST_MSG.format(
                fields=",".join(Digest.model_fields.keys()),
                kind="blog",
                text=text[:DEFAULT_MAX_TEXT_CHARS],
            ),
        },
    ]

def load_inputs(count=32):
    return [create_msg(item['text']) for item in random.sample(load_data_from_directory("data/digests/*.json"), count)]


def load_structured_output_inputs(count=32):
    return [create_structured_output_msg(item['text']) for item in random.sample(load_data_from_directory("data/digests/*.json"), count)]


def parse_structured_output_text(text: str, model_type: Type[BaseModel]) -> BaseModel:
    cleaned = _strip_json_fences(text)
    return model_type.model_validate_json(cleaned)


def parse_tool_call_text(text: str, model_type: Type[BaseModel]) -> BaseModel:
    cleaned = _strip_json_fences(text)
    payload = json.loads(cleaned)

    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"Expected exactly one tool call, got {len(payload)}")
        payload = payload[0]

    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected tool call payload type: {type(payload).__name__}")

    if payload.get("name") == TOOL_NAME:
        arguments = payload.get("arguments", {})
    elif payload.get("function", {}).get("name") == TOOL_NAME:
        arguments = payload["function"].get("arguments", {})
    elif payload.get("tool_name") == TOOL_NAME:
        arguments = payload.get("arguments", {})
    else:
        raise ValueError(f"Could not find {TOOL_NAME} tool call in payload: {payload}")

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    return model_type.model_validate(arguments)


def serialize_outputs(outputs: list[Optional[Digest]]) -> list[Any]:
    return [_safe_model_dump(item) for item in outputs]


class _BaseDigestor:
    def __init__(self, model_name, max_tokens=32768, output_model: Type[BaseModel] = Digest, **sampling_params):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.output_model = output_model
        self.sampling_params = {**sampling_params, "max_tokens": max_tokens}
        self._llm = None
        self._sampling_params = None

    def _ensure_llm(self):
        raise NotImplementedError

    def _parse_output(self, text: str):
        raise NotImplementedError

    def _run_batch(self, messages):
        outputs = self._ensure_llm().chat(messages, sampling_params=self._sampling_params, use_tqdm=True)
        results = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            try:
                results.append(self._parse_output(text))
            except Exception:
                results.append(None)
        return results


class DigestorStructuredOutput(_BaseDigestor):
    def _ensure_llm(self):
        if self._llm is None:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams

            self._llm = LLM(model=self.model_name)
            self._sampling_params = SamplingParams(
                **self.sampling_params,
                structured_outputs=StructuredOutputsParams(
                    json=self.output_model.model_json_schema(),
                    disable_any_whitespace=True,
                ),
            )
        return self._llm

    def _parse_output(self, text: str):
        return parse_structured_output_text(text, self.output_model)

    def run(self, messages):
        return self._run_batch(messages)


class DigestorToolCall(_BaseDigestor):
    def __init__(self, model_name, max_tokens=32768, output_model: Type[BaseModel] = Digest, **sampling_params):
        super().__init__(model_name, max_tokens=max_tokens, output_model=output_model, **sampling_params)
        self._chat_tools = build_chat_tools(self.output_model)

    def _ensure_llm(self):
        if self._llm is None:
            from vllm import LLM, SamplingParams

            self._llm = LLM(model=self.model_name)
            self._sampling_params = SamplingParams(**self.sampling_params)
        return self._llm

    def _parse_output(self, text: str):
        return parse_tool_call_text(text, self.output_model)

    def _run_batch(self, messages):
        outputs = self._ensure_llm().chat(messages, sampling_params=self._sampling_params, tools=self._chat_tools, use_tqdm=True)
        results = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            try:
                results.append(self._parse_output(text))
            except Exception:
                results.append(None)
        return results

    def run(self, messages):
        return self._run_batch(messages)


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

def main_vllm_structured_output(model_name, sampling_params, messages):
    digestor = DigestorStructuredOutput(model_name, **sampling_params)
    outputs = digestor.run(messages)
    save_outputs(serialize_outputs(outputs))


def main_vllm_tool_call(model_name, sampling_params, messages):
    digestor = DigestorToolCall(model_name, **sampling_params)
    return [ic(item) for item in digestor.run(messages)]
    
def save_outputs(outputs):   
    os.makedirs(".test", exist_ok=True)
    with open(f".test/outputs-{datetime.now().strftime('%H-%M-%S')}.json", "w") as f:
        # f.write("[\n")
        # f.write(",\n".join(outputs))
        # f.write("\n]")
        json.dump(outputs, f, indent=2)


# MODEL_NAME = "Qwen/Qwen3.5-0.8B" --> not so good
# MODEL_NAME = "Qwen/Qwen3.6-35B-A3B" 
# sampling_params = SamplingParams(temperature=1.0, top_p=1.0, top_k=40, min_p=0.0, presence_penalty=2.0, repetition_penalty=1.0, max_tokens=32768)
model_name = "LiquidAI/LFM2.5-1.2B-Instruct" # --> great!
sampling_params = {"temperature": 0.35, "top_k": 60, "top_p": 0.95, "repetition_penalty": 1.05, "max_tokens": 32768}
# MODEL_NAME = "LiquidAI/LFM2-1.2B-Extract" --> gets stuck with vllm
# MODEL_NAME = "LiquidAI/LFM2-350M-Extract" --> gets stuck with vllm, but works with transformers generate
if __name__=="__main__":
    messages = load_structured_output_inputs(count=4096)
    res = main_vllm_structured_output(model_name, sampling_params, messages)
    # messages = load_inputs(count=64)
    # res = main_vllm_tool_call(model_name, sampling_params, messages)
    # res = main_transformers(model_name, sampling_params, messages)
    

