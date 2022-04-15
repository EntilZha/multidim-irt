from pathlib import Path
from py_irt.cli import train


def main():
    task = "sentiment"
    out_path = Path("data") / "nn-irt" / task
    data_path = Path("data") / "irt-inputs" / task / "irt-inputs.jsonlines"
    train("nn_2pl", data_path, out_path, config_path="nn_multidim.toml")


if __name__ == "__main__":
    main()
