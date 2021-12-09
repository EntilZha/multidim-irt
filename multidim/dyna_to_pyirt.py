from typing import List
import typer
import pydantic
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd
from pedroai.io import read_json, read_jsonlines, write_json, write_jsonlines
from pedroai.altair import save_chart
import altair as alt
from dataset import Dataset
from multidim.config import conf, DATA_ROOT


app = typer.Typer()
alt.data_transformers.disable_max_rows()


class DynabenchModel(pydantic.BaseModel):
    name: str
    sentiment_dev: str


def list_models() -> List[DynabenchModel]:
    models = []
    for p in Path("data/dynaboard_model_outputs").iterdir():
        if p.is_dir():
            models.append(DynabenchModel(sentiment_dev=str(p), name=p.name))
    return models


def load_gold_labels(task: str):
    dataset_labels = {}
    for dataset, path in conf[task]["dev"]["names"].items():
        labels = {}
        for item in read_jsonlines(DATA_ROOT / path):
            labels[str(item["uid"])] = item["label"]

        dataset_labels[dataset] = labels
    item_to_dataset = {}
    for dataset, labels in dataset_labels.items():
        for item_id in labels.keys():
            item_to_dataset[item_id] = dataset
    return dataset_labels, item_to_dataset


@app.command()
def convert(task: str, output_dir: str):
    models = list_models()
    dataset_labels, _ = load_gold_labels(task)
    model_scores = {m.name: {} for m in models}
    item_correct = defaultdict(int)
    item_total = defaultdict(int)
    datasets = conf[task]["dev"]["names"].keys()
    for m in models:
        for d in datasets:
            for pred in read_jsonlines(
                f"data/dynaboard_model_outputs/{m.name}/{task}/{d}.jsonl.out"
            ):
                item_id = str(pred["id"])
                pred_label = pred["label"]
                gold_label = dataset_labels[d][item_id]
                model_scores[m.name][item_id] = int(pred_label == gold_label)
                item_correct[item_id] += int(pred_label == gold_label)
                item_total[item_id] += 1

    output = []
    for model_name, responses in model_scores.items():
        output.append({"subject_id": model_name, "responses": responses})

    write_jsonlines(Path(output_dir) / "irt-inputs.jsonlines", output)

    item_accuracies = {
        item_id: item_correct[item_id] / item_total[item_id]
        for item_id in item_total.keys()
    }


def load_irt_df(param_file: str, dataset_file: str):
    dataset_labels, item_to_dataset = load_gold_labels()
    params = read_json(param_file)
    dataset = Dataset.from_jsonlines(dataset_file)
    item_accuracies = dataset.get_item_accuracies()
    ix_to_item_id = {int(k): v for k, v in params["item_ids"].items()}
    ix_to_subject_id = {int(k): v for k, v in params["subject_ids"].items()}
    print(ix_to_subject_id)
    print(params["ability"])
    n_items = len(params["diff"])
    rows = []
    for ix in range(n_items):
        item_id = ix_to_item_id[ix]
        diff = params["diff"][ix]
        disc = params["disc"][ix]
        if "lambdas" in params:
            lambda_ = params["lambdas"][ix]
        else:
            lambda_ = np.nan
        dataset = item_to_dataset[item_id]
        rows.append(
            {
                "item_id": item_id,
                "diff": diff,
                "disc": disc,
                "lambda": lambda_,
                "dataset": dataset,
                "accuracy": item_accuracies[item_id].accuracy,
            }
        )
    df = pd.DataFrame(rows)
    df["n"] = 1
    correlations = df.corr(method="kendall")
    print(correlations)
    print(df[df["lambda"] >= 0.5].corr(method="kendall"))

    # Only care about big differences
    if abs(correlations["diff"]["accuracy"]) < 0.1:
        print("diff/accuracy correlations are low, please check the model")

    if correlations["diff"]["accuracy"] > 0:
        print("Signs look ok")
    else:
        print("Signs looked flipped, lets fix that!")
        df["diff"] = -df["diff"]
        df["disc"] = -df["disc"]
    return df


def compute_cdf(dataframe: pd.DataFrame, column: str):
    frames = []
    for _, group in dataframe.set_index("item_id").groupby("dataset"):
        cumsum = group.sort_values(column)["n"].cumsum()
        total = len(group)
        group["cumulative"] = cumsum / total
        frames.append(group)
    return pd.concat(frames)


@app.command()
def analyze(param_file: str, dataset_file: str, output_dir: str):
    output_dir = Path(output_dir)
    df = load_irt_df(param_file, dataset_file)
    scatter = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=alt.X("diff"),
            y=alt.Y("disc"),
            color=alt.Color("dataset"),
            # shape=alt.Shape("dataset"),
        )
    )

    save_chart(scatter, output_dir / "irt_scatter", ["pdf"])

    diff_df = compute_cdf(df[df["lambda"] >= 0.5], "diff")
    diff_dens = (
        alt.Chart(diff_df)
        .mark_area(opacity=0.3, interpolate="step")
        .encode(
            x=alt.X("diff"),
            y=alt.Y("cumulative:Q", stack=None),
            color=alt.Color("dataset"),
        )
    )
    diff_line = (
        alt.Chart(diff_df)
        .mark_line()
        .encode(x=alt.X("diff"), y=alt.Y("cumulative"), color=alt.Color("dataset"))
    )
    save_chart(diff_dens + diff_line, output_dir / "irt_diff_dens", ["pdf"])

    disc_df = compute_cdf(df[df["lambda"] >= 0.5], "disc")
    disc_dens = (
        alt.Chart(disc_df)
        .mark_area(opacity=0.3, interpolate="step")
        .encode(
            x=alt.X("disc"),
            y=alt.Y("cumulative:Q", stack=None),
            color=alt.Color("dataset"),
        )
    )
    disc_line = (
        alt.Chart(disc_df)
        .mark_line()
        .encode(x=alt.X("disc"), y=alt.Y("cumulative"), color=alt.Color("dataset"))
    )
    save_chart(disc_dens + disc_line, output_dir / "irt_disc_dens", ["pdf"])

    if not df["lambda"].isnull().any():
        lambda_df = compute_cdf(df, "lambda")
        lambda_dens = (
            alt.Chart(lambda_df)
            .mark_area(opacity=0.3, interpolate="step")
            .encode(
                x=alt.X("lambda"),
                y=alt.Y("cumulative:Q", stack=None),
                color=alt.Color("dataset"),
            )
        )
        lambda_line = (
            alt.Chart(lambda_df)
            .mark_line()
            .encode(
                x=alt.X("lambda"), y=alt.Y("cumulative"), color=alt.Color("dataset")
            )
        )
        save_chart(lambda_dens + lambda_line, output_dir / "irt_lambda_dense", ["pdf"])


if __name__ == "__main__":
    app()
