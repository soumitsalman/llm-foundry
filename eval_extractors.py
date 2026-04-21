from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()


def batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


from gliner import GLiNER
from icecream import ic

from utils import count_words, load_data_from_directory

LABELS = [
    "person",
    "organization",
    "company",
    "institution",
    "business",
    "city",
    "state",
    "country",
    "stock",
    "ticker",
    "stockticker",
    # "organization_company",
    # "city_state_country",
    "product",
    # "business_stock_ticker",
]
WORDS_THRESHOLD = 500
CONTEXT_LEN = 2048

beans = load_data_from_directory(
    "data/digests/*",
    lambda bean: count_words(bean["text"]) <= WORDS_THRESHOLD,
)
model = GLiNER.from_pretrained(
    "knowledgator/modern-gliner-bi-base-v1.0",
    max_length=CONTEXT_LEN,
    map_location="cuda",
)


for batch in batched(beans, 8):
    entities = model.inference(
        [bean["text"] for bean in batch],
        labels=LABELS,
        threshold=0.3,
        max_length=CONTEXT_LEN,
        batch_size=2,
    )
    for group in entities:
        digest = defaultdict(set)
        for entity in group:
            digest[entity["label"]].add(entity["text"].lower())
        ic(digest)
