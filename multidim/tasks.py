from collections import defaultdict
from pathlib import Path
import luigi
from pedroai.io import read_jsonlines, write_jsonlines
from multidim.dyna_to_pyirt import list_models, load_gold_labels
from multidim.config import DATA_ROOT, conf
from multidim.topic import TopicModel, task_data_to_mallet
from multidim import dyna_to_pyirt
from multidim.log import get_logger
from py_irt.cli import train


log = get_logger(__name__)

DB_TASKS = ["sentiment"]


class DBTaskDataset(luigi.ExternalTask):
    task = luigi.Parameter()
    dataset = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(
            DATA_ROOT / conf[self.task]["dev"]["data"][self.dataset]
        )


class DBMergedTaskDataset(luigi.Task):
    task = luigi.Parameter()

    @property
    def _output_path(self):
        return DATA_ROOT / "datasets" / self.task / "merged.jsonl"

    def requires(self):
        for dataset in conf[self.task]["dev"]["data"].keys():
            yield DBTaskDataset(task=self.task, dataset=dataset)

    def run(self):
        examples = []
        for path in conf[self.task]["dev"]["data"].values():
            task_examples = read_jsonlines(DATA_ROOT / path)
            log.info(f"Reading {self.task}: {len(task_examples)} examples")
            examples.extend(task_examples)
        write_jsonlines(self._output_path, examples)

    def output(self):
        return luigi.LocalTarget(self._output_path)


class FormatMergedTaskData(luigi.Task):
    task = luigi.Parameter()

    def requires(self):
        return DBMergedTaskDataset(task=self.task)

    @property
    def _input_path(self):
        return DATA_ROOT / "datasets" / self.task / "merged.jsonl"

    @property
    def _output_path(self):
        return DATA_ROOT / "mallet" / self.task / "merged.mallet"

    def run(self):
        log.info("Reformatting data from jsonl to mallet")
        log.info(f"Input: {self._input_path}")
        log.info(f"Output: {self._output_path}")
        task_data_to_mallet(self.task, self._input_path, self._output_path)

    def output(self):
        return self._output_path


class FormatTaskDatasetData(luigi.Task):
    task = luigi.Parameter()
    dataset = luigi.Parameter()

    def requires(self):
        return DBTaskDataset(task=self.task, dataset=self.dataset)

    @property
    def _input_path(self):
        return DATA_ROOT / conf[self.task]["dev"]["data"][self.dataset]

    @property
    def _output_path(self):
        return DATA_ROOT / conf[self.task]["dev"]["mallet"][self.dataset]

    def run(self):
        log.info("Reformatting data from jsonl to mallet")
        log.info(f"Input: {self._input_path}")
        log.info(f"Output: {self._output_path}")
        task_data_to_mallet(self.task, self._input_path, self._output_path)

    def output(self):
        return self._output_path


class PerTaskDatasetTopicModel(luigi.Task):
    task = luigi.Parameter()
    dataset = luigi.Parameter()
    num_topics = luigi.IntParameter(default=10)
    optimize_interval = luigi.IntParameter(default=10)
    remove_stopwords = luigi.BoolParameter(default=True)
    num_iterations = luigi.IntParameter(default=10)
    doc_topics_threshold = luigi.FloatParameter(default=0.05)

    def requires(self):
        return FormatTaskDatasetData(task=self.task, dataset=self.dataset)

    @property
    def _input_path(self):
        return DATA_ROOT / conf[self.task]["dev"]["mallet"][self.dataset]

    @property
    def _output_dir(self):
        return (
            DATA_ROOT
            / conf[self.task]["dev"]["topic"][f"num_topics={self.num_topics}"][
                "output_dir"
            ]
            / self.dataset
        )

    def run(self):
        model = TopicModel(
            input_file=self._input_path,
            num_topics=self.num_topics,
            output_dir=self._output_dir,
            optimize_interval=self.optimize_interval,
            remove_stopwords=self.remove_stopwords,
            num_iterations=self.num_iterations,
            doc_topics_threshold=self.doc_topics_threshold,
        )
        model.train()

    def output(self):
        return [
            luigi.LocalTarget(self._output_dir / "mallet.model"),
            luigi.LocalTarget(self._output_dir / "mallet.state.gz"),
            luigi.LocalTarget(self._output_dir / "mallet.topic_distributions"),
            luigi.LocalTarget(self._output_dir / "mallet.topic_keys"),
        ]


class MergedTaskTopicModel(luigi.Task):
    task = luigi.Parameter()
    num_topics = luigi.IntParameter(default=10)
    optimize_interval = luigi.IntParameter(default=10)
    remove_stopwords = luigi.BoolParameter(default=True)
    num_iterations = luigi.IntParameter(default=10)
    doc_topics_threshold = luigi.FloatParameter(default=0.05)

    def requires(self):
        return FormatMergedTaskData(task=self.task)

    @property
    def input_path(self):
        return (
            DATA_ROOT
            / conf[self.task]["dev"]["topic"][f"num_topics={self.num_topics}"]["mallet"]
        )

    @property
    def output_dir(self):
        return (
            DATA_ROOT
            / conf[self.task]["dev"]["topic"][f"num_topics={self.num_topics}"][
                "output_dir"
            ]
            / "merged"
        )

    def run(self):
        model = TopicModel(
            input_file=self.input_path,
            num_topics=self.num_topics,
            output_dir=self.output_dir,
            optimize_interval=self.optimize_interval,
            remove_stopwords=self.remove_stopwords,
            num_iterations=self.num_iterations,
            doc_topics_threshold=self.doc_topics_threshold,
        )
        model.train()

    def output(self):
        return [
            luigi.LocalTarget(self.output_dir / "mallet.model"),
            luigi.LocalTarget(self.output_dir / "mallet.state.gz"),
            luigi.LocalTarget(self.output_dir / "mallet.topic_distributions"),
            luigi.LocalTarget(self.output_dir / "mallet.topic_keys"),
        ]


class TaskToPyIrt(luigi.Task):
    task = luigi.Parameter()

    def requires(self):
        for dataset in conf[self.task]["dev"]["data"].keys():
            yield DBTaskDataset(task=self.task, dataset=dataset)

    def run(self):
        dataset_labels, _ = load_gold_labels()
        models = list(list_models())
        if len(models) == 0:
            raise ValueError("Number of models is zero")
        model_scores = {m.name: {} for m in models}
        item_correct = defaultdict(int)
        item_total = defaultdict(int)
        task_datasets = conf[self.task]["dev"]["names"].values()
        for m in models:
            for d in task_datasets:
                for pred in read_jsonlines(
                    f"data/dynaboard_model_outputs/{m.name}/{self.task}/{d}.jsonl.out"
                ):
                    item_id = str(pred["id"])
                    pred_label = pred["label"]
                    gold_label = dataset_labels[d][item_id]
                    model_scores[m.name][item_id] = int(pred_label == gold_label)
                    item_correct[item_id] += int(pred_label == gold_label)
                    item_total[item_id] += 1

        output = []
        for model_name, responses in model_scores.items():
            if len(responses) == 0:
                raise ValueError("Zero responses")
            output.append({"subject_id": model_name, "responses": responses})

        output_dir = Path("data/") / "irt-inputs" / self.task
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / "irt-inputs.jsonlines"
        write_jsonlines(output_path, output)

    def output(self):
        return Path("data/") / "irt-inputs" / self.task / "irt-inputs.jsonlines"


class TrainPyIrt(luigi.Task):
    task = luigi.Parameter()

    @property
    def _output_dir(self):
        return Path("data") / "irt-models" / self.task

    def requires(self):
        yield TaskToPyIrt(task=self.task)

    def run(self):
        data_path = Path("data/") / "irt-inputs" / self.task / "irt-inputs.jsonlines"
        train("2pl", data_path, self._output_dir, config_path="multidim.toml")

    def output(self):
        return [
            self._output_dir / "parameters.json",
            self._output_dir / "best_parameters.json",
        ]


class AllTasks(luigi.WrapperTask):
    def requires(self):
        for task in DB_TASKS:
            yield TrainPyIrt(task=task)
            yield MergedTaskTopicModel(task=task)
            for dataset in conf[task]["dev"]["names"].keys():
                yield PerTaskDatasetTopicModel(task=task, dataset=dataset)
