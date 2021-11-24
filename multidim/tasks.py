from collections import defaultdict
from multiprocessing import Value
from pathlib import Path
import luigi
from pedroai.io import read_jsonlines, write_jsonlines
from multidim.dyna_to_pyirt import list_models, load_gold_labels
from multidim.config import DATA_ROOT, conf
from multidim.topic import TopicModel, reviews_to_mallet
from multidim import dyna_to_pyirt
from multidim.log import get_logger


log = get_logger(__name__)


class SentimentData(luigi.ExternalTask):
    dataset = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(DATA_ROOT / conf["reviews"]["dev"][self.dataset])


class FormatSentimentData(luigi.Task):
    dataset = luigi.Parameter()

    def requires(self):
        return SentimentData(dataset=self.dataset)

    @property
    def _input_path(self):
        return DATA_ROOT / conf["mallet_reviews"]["dev"][self.dataset]

    @property
    def _output_path(self):
        return DATA_ROOT / conf["mallet_reviews"]["dev"][self.dataset]

    def run(self):
        log.info("Reformatting data from jsonl to mallet")
        log.info(f"Input: {self._input_path}")
        log.info(f"Output: {self._output_path}")
        reviews_to_mallet(self._input_path, self._output_path)

    def output(self):
        return self._output_path


class SentimentTopicModel(luigi.Task):
    dataset = luigi.Parameter()
    num_topics = luigi.IntParameter(default=10)
    optimize_interval = luigi.IntParameter(default=10)
    remove_stopwords = luigi.BoolParameter(default=True)
    num_iterations = luigi.IntParameter(default=10)
    doc_topics_threshold = luigi.FloatParameter(default=0.05)

    def requires(self):
        return FormatSentimentData(dataset=self.dataset)

    @property
    def _input_path(self):
        return DATA_ROOT / conf["mallet_reviews"]["dev"][self.dataset]

    @property
    def _output_dir(self):
        return DATA_ROOT / conf["topic"][f"num_topics={self.num_topics}"]["output_dir"]

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


class AllSentimentExperiments(luigi.WrapperTask):
    def requires(self):
        for dataset in conf["reviews"]["dev"].keys():
            yield SentimentTopicModel(dataset=dataset)


class SentimentToPyIrt(luigi.Task):
    task = luigi.Parameter(default="sentiment")

    def requires(self):
        for dataset in conf["reviews"]["dev"].keys():
            yield SentimentData(dataset=dataset)

    def run(self):
        dataset_labels, _ = load_gold_labels()
        models = list(list_models())
        if len(models) == 0:
            raise ValueError("Number of models is zero")
        model_scores = {m.name: {} for m in models}
        item_correct = defaultdict(int)
        item_total = defaultdict(int)
        for m in models:
            for d in dyna_to_pyirt.sentiment_datasets:
                for pred in read_jsonlines(
                    f"data/dynaboard_model_outputs/{m.name}/{self.task}/{d}.jsonl.out"
                ):
                    item_id = str(pred["id"])
                    pred_label = pred["label"]
                    gold_label = dataset_labels[d][item_id]
                    model_scores[m.name][item_id] = int(pred_label == gold_label)
                    item_correct[item_id] += int(pred_label == gold_label)
                    item_total[item_id] += 1

        breakpoint
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
