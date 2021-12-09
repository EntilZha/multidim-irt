import abc
from typing import List
from enum import Enum
import pydantic


class SentimentLabel(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class ExtractText(abc.ABC):
    @abc.abstractmethod
    def example_text(self) -> str:
        pass


class Sentiment(pydantic.BaseModel, ExtractText):
    uid: str
    statement: str
    label: SentimentLabel

    def example_text(self) -> str:
        return self.statement

    @classmethod
    def from_jsonlines(cls, path: str) -> List["Sentiment"]:
        with open(path) as f:
            rows = []
            for line in f:
                rows.append(cls.parse_raw(line))
            return rows


class NliLabel(str, Enum):
    contradictory = "contradictory"
    neutral = "neutral"
    entailed = "entailed"


class Inference(pydantic.BaseModel, ExtractText):
    uid: str
    context: str
    hypothesis: str
    label: NliLabel

    def example_text(self) -> str:
        return f"{self.context} {self.hypothesis}"

    @classmethod
    def from_jsonlines(cls, path: str) -> List["Inference"]:
        with open(path) as f:
            rows = []
            for line in f:
                rows.append(cls.parse_raw(line))
            return rows
