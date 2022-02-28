import pandas as pd
import altair as alt
from multidim.dynaboard_datasets import Sentiment, NliLabel
from multidim.config import conf, DATA_ROOT
from multidim.topic import TopicModelOutput
from multidim.tasks import MergedTaskTopicModel, TopicModel


from typing import Union, List, Optional
from pathlib import Path
import random
import altair as alt
from rich.console import Console
import altair_saver
import typer

from pedroai.io import read_json, requires_file, safe_file
from pedroai.math import to_precision


alt.data_transformers.disable_max_rows()

console = Console()

PAPER_PATH = Path("data/paper-figures")
AUTO_FIG = PAPER_PATH / "auto_fig"
COMMIT_AUTO_FIG = PAPER_PATH / "commit_auto_fig"


def save_chart(chart: alt.Chart, base_path: Union[Path, str], filetypes: List[str]):
    base_path = str(base_path)
    for t in filetypes:
        path = base_path + "." + t
        if t in ("svg", "pdf"):
            method = "node"
        else:
            method = None
        altair_saver.save(chart, safe_file(path), method=method)


PLOTS = {}


def register_plot(name: str):
    def decorator(func):
        PLOTS[name] = func
        return func

    return decorator


@register_plot("topics_table")
def latex_topics_table(filetypes: List[str], commit: bool = False):
    task = "sentiment"
    num_topics = 5
    rows = []
    topic_file = (
        DATA_ROOT
        / conf[task]["dev"]["topic"][f"num_topics={num_topics}"]["output_dir"]
        / "merged"
        / "mallet.topic_keys"
    )
    with open(topic_file) as f:
        for line in f:
            topic_id, score, words = line.strip().split("\t")
            words = " ".join([w for w in words.split()][:10])
            rows.append({"topic_id": topic_id, "score": score, "words": words})
    topic_df = pd.DataFrame(rows)
    with pd.option_context("max_colwidth", 1000):
        table = topic_df.to_latex(
            columns=["topic_id", "words"],
            header=["Topic ID", "Topic Words in Dynabench Sentiment Datasets"],
            index=False,
        )
    with open(AUTO_FIG / "sentiment_topics.tex", "w") as f:
        f.write(table)


@register_plot("diff_by_topic")
def plot_diff_by_topic(filetypes: List[str], commit: bool = False):
    sentiment_datasets = {}
    task = "sentiment"

    for name, path in conf[task]["dev"]["data"].items():
        sentiment_datasets[name] = Sentiment.from_jsonlines(DATA_ROOT / path)
    num_topics = 5
    tm_task = MergedTaskTopicModel(task=task, num_topics=num_topics)
    assert tm_task.complete()
    model = TopicModelOutput(tm_task.output_dir)
    irt_params = read_json(DATA_ROOT / "nn-irt" / "sentiment" / "best_parameters.json")
    id_to_diff = {}
    for str_idx, item_id in irt_params["item_ids"].items():
        idx = int(str_idx)
        id_to_diff[item_id] = irt_params["diff"][idx]
    id_to_example = {}
    for examples in sentiment_datasets.values():
        for ex in examples:
            id_to_example[ex.uid] = ex
    id_to_topic = {}
    for doc in model.topic_distribution.values():
        id_to_topic[doc["example_id"]] = doc["top_topic_id"]
    rows = []
    for uid in id_to_example.keys():
        example = id_to_example[uid]
        rows.append(
            {
                "diff": id_to_diff[uid],
                "top_topic": id_to_topic[uid],
                "text": example.example_text(),
            }
        )
    df = pd.DataFrame(rows)
    width = 600
    bins = 25
    hist = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("diff", bin=alt.Bin(maxbins=bins), title=None),
            y=alt.Y("count()", title="Count"),
        )
        .properties(height=40, width=width)
    )
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("diff", bin=alt.Bin(maxbins=bins), title="IRT Difficulty"),
            y=alt.Y("count()", stack="normalize", title="Percent of Histogram Bin"),
            color=alt.Color(
                "top_topic:N",
                title="Example Topic",
                legend=alt.Legend(
                    orient="none",
                    titleAnchor="middle",
                    legendX=225,
                    legendY=-42,
                    direction="horizontal",
                ),
            ),
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(dy=8, color="white", fontSize=14)
        .encode(
            x=alt.X("diff", bin=alt.Bin(maxbins=bins)),
            y=alt.Y("count()", stack="normalize"),
            detail="top_topic:N",
            text=alt.Text("count()"),
        )
    )
    chart = hist & (bars + text).properties(width=width)
    chart = chart.configure_axis(titleFontSize=18, labelFontSize=16).configure_legend(
        labelFontSize=16, titleFontSize=18
    )
    save_chart(chart, AUTO_FIG / "diff_by_topic", filetypes)


def main(
    plot: Optional[List[str]] = typer.Option(None),
    seed: int = 42,
    filetype: Optional[List[str]] = typer.Option(None),
    commit: bool = False,
):
    random.seed(seed)
    if filetype is None or len(filetype) == 0:
        filetype = ["svg", "pdf", "json", "png"]

    console.log("Output Filetypes:", filetype)
    console.log("Commit:", commit)

    if plot is None or len(plot) == 0:
        for name, func in PLOTS.items():
            console.log(f"Plotting: {name}")
            func(filetype, commit=commit)
    else:
        for p in plot:
            console.log(f"Plotting: {p}")
            PLOTS[p](filetype, commit=commit)


if __name__ == "__main__":
    typer.run(main)
