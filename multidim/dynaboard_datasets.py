from typing import List
from enum import Enum
import pydantic


class LabelEnum(str, Enum):
    positive = 'positive'
    neutral = 'neutral'
    negative = 'negative'


class AmazonReview(pydantic.BaseModel):
    uid: str
    statement: str
    label: LabelEnum

    @classmethod
    def from_jsonlines(cls, path: str) -> List['AmazonReview']:
        with open(path) as f:
            rows = []
            for line in f:
                rows.append(cls.parse_raw(line))
            return rows
