"""Copyright (c) Facebook, Inc. and its affiliates."""
from typing import List
import re
from pathlib import Path
from typing import Union

import numpy as np
import typer
from pedroai.io import read_json, shell, write_json
from rich.console import Console
from pedroai.io import safe_file

from multidim.dynaboard_datasets import Inference, Sentiment

console = Console()


topic_app = typer.Typer()
STOPS = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "ve",
    "ll",
    "amp",
]


def load_topics(file: str):
    uid_to_topics = {}
    with open(file) as f:
        for line in f:
            tokens = line.split()
            uid = tokens[1]
            probs = [float(p) for p in tokens[2:]]
            topic_id = np.argmax(probs)
            uid_to_topics[uid] = (topic_id, probs)
    return uid_to_topics


PROCESSED_MALLET_FILENAME = "input_data.mallet"


def process_string(
    text,
    lowercase=True,
    remove_short_words=True,
    remove_stop_words=True,
    remove_punctuation=True,
    numbers="replace",
    stop_words=STOPS,
    stop_words_extra=[],
):
    if lowercase:
        text = text.lower()
    if numbers == "replace":
        text = re.sub("[0-9]+", "NUM", text)
    elif numbers == "remove":
        text = re.sub("[0-9]+", " ", text)
    if remove_punctuation:
        text = re.sub(r"[^\sA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ/]", " ", text)
    if remove_stop_words:
        text = " ".join(
            [word for word in text.split() if word not in stop_words + stop_words_extra]
        )
    if remove_short_words:
        text = " ".join([word for word in text.split() if not len(word) <= 2])
    text = " ".join(text.split())
    return text


class TopicModel:
    def __init__(
        self,
        *,
        model_dir: Union[str, Path],
        num_topics: int,
        optimize_interval: int = 10,
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        remove_short_words: bool = True,
        lowercase: bool = True,
        num_iterations: int = 1000,
        doc_topics_threshold: float = 0.05,
        random_seed: int = 0,
    ) -> None:
        super().__init__()
        self._model_dir = model_dir
        self._input_file = Path(model_dir) / PROCESSED_MALLET_FILENAME
        self._num_topics = num_topics
        self._optimize_interval = optimize_interval
        self._output_dir = Path(model_dir)
        self._output_dir.mkdir(exist_ok=True, parents=True)
        self._remove_stopwords = remove_stopwords
        self._num_iterations = num_iterations
        self._random_seed = random_seed
        self._doc_topics_threshold = doc_topics_threshold
        self._lowercase = lowercase
        self._remove_punctuation = remove_punctuation
        self._remove_short_words = remove_short_words

    def preprocess(self, task_name: str, input_json_file: str) -> List:
        console.log(f"Reformatting data from jsonl to mallet for {task_name}")
        console.log(f"Input: {input_json_file}")
        console.log(f"Output: {self._input_file}")
        if task_name == "sentiment":
            data = Sentiment.from_jsonlines(input_json_file)
        elif task_name == "nli":
            data = Inference.from_jsonlines(input_json_file)
        else:
            raise NotImplementedError()

        with open(safe_file(self._input_file), "w") as f:
            for r in data:
                text = process_string(
                    r.example_text(),
                    lowercase=self._lowercase,
                    remove_short_words=self._remove_short_words,
                    remove_stop_words=self._remove_stopwords,
                    remove_punctuation=self._remove_punctuation,
                )
                f.write(f"{r.uid}\t{r.label}\t{text}\n")

    @classmethod
    def load(cls, output_dir: Union[str, Path]):
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        parameters = read_json(output_dir / "parameters.json")
        return cls(
            input_file=parameters["input_file"],
            num_topics=parameters["num_topics"],
            output_dir=parameters["output_dir"],
            optimize_interval=parameters["optimize_interval"],
            remove_stopwords=parameters["remove_stopwords"],
            doc_topics_threshold=parameters["doc_topics_threshold"],
            random_seed=parameters["random_seed"],
        )

    def import_data_command(self):
        args = ["mallet", "import-file", "--keep-sequence"]
        if self._remove_stopwords:
            args.append("--remove-stopwords")
        if not self._lowercase:
            args.append("--preserve-case")
        args.append("--input")
        args.append(str(self._input_file))
        args.append("--output")
        args.append(str(self._output_dir / PROCESSED_MALLET_FILENAME))
        command = " ".join(args)
        return command

    def _import_data(self):
        command = self.import_data_command()
        console.log("Running: ", command)
        shell(command)

    @property
    def doc_topics_file(self):
        return str(self._output_dir / "mallet.topic_distributions")

    @property
    def topic_keys_file(self):
        return str(self._output_dir / "mallet.topic_keys")

    @property
    def model_state_file(self):
        return str(self._output_dir / "mallet.state.gz")

    def train_command(self):
        args = [
            "mallet",
            "train-topics",
            "--input",
            str(self._output_dir / PROCESSED_MALLET_FILENAME),
            "--num-topics",
            str(self._num_topics),
            "--output-topic-keys",
            self.topic_keys_file,
            "--output-doc-topics",
            self.doc_topics_file,
            "--inferencer-filename",
            str(self._output_dir / "mallet.model"),
            "--output-state",
            self.model_state_file,
            "--random-seed",
            str(self._random_seed),
            "--doc-topics-threshold",
            str(self._doc_topics_threshold),
            "--optimize-interval",
            str(self._optimize_interval),
            "--num-iterations",
            str(self._num_iterations),
        ]
        command = " ".join(args)
        return command

    def train(self):
        self._import_data()
        command = self.train_command()
        console.log("Running: ", command)
        shell(command)
        write_json(
            self._output_dir / "parameters.json",
            {
                "input_file": str(self._input_file),
                "num_topics": self._num_topics,
                "output_dir": str(self._output_dir),
                "optimize_interval": self._optimize_interval,
                "remove_stopwords": self._remove_stopwords,
                "doc_topics_threshold": self._doc_topics_threshold,
                "random_seed": self._random_seed,
            },
        )


def parse_topic_line(line: str):
    entries = line.strip().split("\t")
    doc_id = int(entries[0])
    example_id = entries[1]
    proportions = entries[2:]
    topic_dist = {}
    for i in range(0, len(proportions), 2):
        topic_id = int(proportions[i])
        prob = float(proportions[i + 1])
        topic_dist[topic_id] = prob
    return {
        "doc_id": doc_id,
        "example_id": example_id,
        "topic_dist": topic_dist,
        "top_topic_id": int(proportions[0]),
    }


def parse_topic_distributions(input_dir: str):
    results = {}
    with open(input_dir) as f:
        for line in f:
            if line.startswith("#doc"):
                continue
            parsed = parse_topic_line(line)
            if parsed["doc_id"] in results:
                raise ValueError()
            results[parsed["doc_id"]] = parsed
    return results


class TopicModelOutput:
    def __init__(self, input_dir: str) -> None:
        self.input_dir = input_dir
        self.topic_distribution = parse_topic_distributions(
            Path(input_dir) / "mallet.topic_distributions"
        )


@topic_app.command()
def train(
    input_file: str,
    output_dir: Path,
    optimize_interval: int = 10,
    topics: int = 10,
    remove_stopwords: bool = True,
    num_iterations: int = 1000,
    random_seed: int = 0,
):
    model = TopicModel(
        input_file=input_file,
        num_topics=topics,
        output_dir=output_dir,
        optimize_interval=optimize_interval,
        remove_stopwords=remove_stopwords,
        num_iterations=num_iterations,
        random_seed=random_seed,
    )
    model.train()
