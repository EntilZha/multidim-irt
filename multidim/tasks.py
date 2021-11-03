import luigi
from multidim.config import DATA_ROOT, conf
from multidim.topic import TopicModel, reviews_to_mallet
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
