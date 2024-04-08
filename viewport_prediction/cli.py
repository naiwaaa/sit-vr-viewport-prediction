from pathlib import Path
from argparse import ArgumentParser

from viewport_prediction import __version__
from viewport_prediction.models import ALL_MODELS
from viewport_prediction.experiment import run_experiment


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_experiment(model_name=args.model_name, config_file=args.config_file)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="viewport_prediction", description="Run experiment")

    parser.add_argument(
        "-m",
        "--model",
        dest="model_name",
        metavar="MODEL_NAME",
        help=f"Model name {list(ALL_MODELS.keys())}",
        required=True,
        type=str,
        choices=ALL_MODELS.keys(),
    )

    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        metavar="FILE",
        help="Config file location",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print version info",
    )

    return parser
