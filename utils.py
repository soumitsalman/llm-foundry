import glob
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import tiktoken
from icecream import ic
from retry import retry

log = logging.getLogger("app")

MAX_TOKENS = 6144
# db = MongoClient(os.getenv("LOCAL_DB"), retryWrites=True)["trainingdata"]
encoding = tiktoken.get_encoding("cl100k_base")

current_time = lambda: datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
truncate = lambda text: encoding.decode(encoding.encode(text)[:MAX_TOKENS])


def count_words(text: str) -> int:
    return len(text.split())


def batch_truncate(beans: list):
    tokenlist = encoding.encode_batch(
        [bean["text"] for bean in beans], num_threads=os.cpu_count()
    )
    tokenlist = [tokens[:MAX_TOKENS] for tokens in tokenlist]
    texts = encoding.decode_batch(tokenlist, num_threads=os.cpu_count())
    for bean, text in zip(beans, texts):
        bean["text"] = text
    return beans


def pad_current_time(items: list[dict], use_str=True):
    now = current_time() if use_str else datetime.now()
    [item.update({"collected": now}) for item in items]
    return items


def save_data_to_file(items, what):
    if not items:
        return

    os.makedirs(".data", exist_ok=True)
    filename = (
        ".data/" + re.sub(r"[^a-zA-Z0-9]", "-", f"{what}-{current_time()}") + ".json"
    )
    with open(filename, "w") as file:
        json.dump(pad_current_time(items), file)


def load_data_from_file_path(file_path):
    with open(file_path, "r") as file:
        items = json.load(file)
    return items


def load_data_from_directory(file_name_prefix, filter_func=lambda bean: True):
    beans = []
    for file_path in glob.glob(file_name_prefix):
        beans.extend(filter(filter_func, load_data_from_file_path(file_path)))
    return beans


def save_data_to_file_path(items, file_path):
    if not items:
        return
    with open(file_path, "w") as file:
        json.dump(items, file)


def save_jsonl_to_file_path(items, file_path):
    if not items:
        return
    with open(file_path, "w") as file:
        file.writelines([json.dumps(row) + "\n" for row in items])


def save_data_to_directory(data: list[dict], directory: str, file_name_prefix: str):
    os.makedirs(directory, exist_ok=True)
    batch_size = 1000
    with ThreadPoolExecutor() as exec:
        exec.map(
            lambda i: save_data_to_file_path(
                data[i : i + batch_size],
                f"{directory}/{file_name_prefix}-{i}-{i + len(data[i : i + batch_size])}.json",
            ),
            range(0, len(data), batch_size),
        )


def save_jsonl_to_directory(data: list[dict], directory: str, file_name_prefix: str):
    os.makedirs(directory, exist_ok=True)
    batch_size = 1000
    with ThreadPoolExecutor() as exec:
        exec.map(
            lambda i: save_jsonl_to_file_path(
                data[i : i + batch_size],
                f"{directory}/{file_name_prefix}-{i}-{i + len(data[i : i + batch_size])}.jsonl",
            ),
            range(0, len(data), batch_size),
        )


def port_data(from_file_pattern: str, to_directory: str, to_prefix: str):
    beans = load_data_from_directory(from_file_pattern)
    collected = int(datetime.now().timestamp())
    for bean in beans:
        bean["collected"] = collected
    save_data_to_directory(beans, to_directory, to_prefix)


def print_results(items, what):
    if not items:
        return
    for item in items:
        print("===========NEW ITEM===============")
        print(item["_id"])
        print(item.get(what))
        print("===========END ITEM===============")


total_beans = 0


def measure_output(func):
    async def wrapper(*args, **kwargs):
        global total_beans

        start = datetime.now()
        result = await func(*args, **kwargs)
        duration = datetime.now() - start

        total_beans += len(result) if isinstance(result, list) else 0
        print(f"[{datetime.now()}] {duration} | {total_beans}")
        return result

    return wrapper
