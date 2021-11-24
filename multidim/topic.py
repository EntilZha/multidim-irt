"""Copyright (c) Facebook, Inc. and its affiliates."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import typer
from pedroai.io import read_json, shell, write_json
from rich.console import Console

from multidim.config import DATA_ROOT, conf
from multidim.dynaboard_datasets import Sentiment

console = Console()


topic_app = typer.Typer()


def reviews_to_mallet(input_file: str, output_file: str):
    reviews = Sentiment.from_jsonlines(input_file)
    with open(output_file, "w") as f:
        for r in reviews:
            f.write(f"{r.uid}\t{r.label}\t{r.statement}\n")


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


class TopicModel:
    PROCESSED_INPUT = "input_data.mallet"

    def __init__(
        self,
        *,
        input_file: Union[str, Path],
        num_topics: int,
        output_dir: Union[str, Path],
        optimize_interval: int = 10,
        remove_stopwords: bool = True,
        num_iterations: int = 1000,
        doc_topics_threshold: float = 0.05,
        random_seed: int = 0,
    ) -> None:
        super().__init__()
        self._input_file = Path(input_file)
        self._num_topics = num_topics
        self._optimize_interval = optimize_interval
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True, parents=True)
        self._remove_stopwords = remove_stopwords
        self._num_iterations = num_iterations
        self._random_seed = random_seed
        self._doc_topics_threshold = doc_topics_threshold

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

    def _import_data(self):
        args = ["mallet", "import-file", "--keep-sequence", "--preserve-case"]
        if self._remove_stopwords:
            args.append("--remove-stopwords")
        args.append("--input")
        args.append(str(self._input_file))
        args.append("--output")
        args.append(str(self._output_dir / self.PROCESSED_INPUT))
        command = " ".join(args)
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

    def train(self):
        self._import_data()
        args = [
            "mallet",
            "train-topics",
            "--input",
            str(self._output_dir / self.PROCESSED_INPUT),
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
        ]
        command = " ".join(args)
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
