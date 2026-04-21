from vllm import LLM
from transformers import AutoTokenizer
from models import *
from utils import *

MODEL_NAME = "unsloth/Qwen3-0.6B-bnb-4bit"
DIGESTOR_SYS = f"""RESPONSE FORMAT:\n```json\n{NewsSummaryBase.model_json_schema()}```\n"""
DIGESTOR_INST = f"""EXTRACT {','.join(NewsSummaryBase.model_fields.keys())} FROM \n```\n# Double Dragon Revive is getting a free bonus game inspired by a retro classic\n\nDouble Dragon Revive, the remake of the classic retro game, is getting a surprising bonus game inspired by one of Technos' most-loved titles. Double Dragon Dodgeball is a revised version of the addictive Super Dodge Ball, and will be free to anyone who pre-orders Double Dragon Revive.\nDouble Dragon Dodgeball doesn't attempt the 3D updating of Revive and instead keeps things traditionally pixel art. The bonus game features 56 characters from the Double Dragon series and features similar arcade-perfect gameplay as Super Dodge Ball. The addition of a Story Mode where dodge-brawlers need to save Marian is fun, while 1v1 and 4v4 multiplayer will appeal to retro fans.\nDouble Dragon Revive has been on my radar for ever since Arc System Works announced it was bringing the 1980s arcade brawler back on the best games consoles. This is a long-overdue 3D remake of a game I wasted far too many hours on in the arcades (and yes, I did use the elbow cheat) and looks to be a substantial revision of the classic side-scrolling brawler.\nThe new Double Dragon Revive update releases 23 October, and you can pre-order now to get your hands on Double Dragon Dodgeball. The only downside is there's currently no Nintendo Switch version in development.\nIf you can't wait for the remake, you can play the original Double Dragon and NES versions on some of the best retro consoles, including an excellent Technos-themed Hyper Mega Tech! Super Pocket edition.\n```"""

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto"
)

# prepare the model input
messages = [
    {"role": "system", "content": DIGESTOR_SYS},
    {"role": "user", "content": DIGESTOR_INST}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
