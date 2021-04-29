import argparse
import json
import sys
from pathlib import Path


def parseConfig(sys_argv=None):
    if sys_argv is None:
        sys_argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-workers',
                        help='Number of worker processes for background data loading',
                        default=8,
                        type=int,
                        )
    parser.add_argument('--batch-size',
                        help='Batch size to use for training',
                        default=32,
                        type=int,
                        )
    parser.add_argument('--epochs',
                        help='Number of epochs to train for',
                        default=1,
                        type=int,
                        )
    parser.add_argument('--validation-size',
                        help='Relative size of the validation split [0-1[',
                        default=0.3,
                        type=float,
                        )
    parser.add_argument('--monitoring-metric',
                        help="What metric to use for best model saving",
                        default='auc',
                        type=str
                        )
    parser.add_argument('--dataset',
                        help="What dataset to feed the model.",
                        action='store',
                        default='LeftRightDataset',
                        )
    parser.add_argument('--experiment-name',
                        default='test',
                        help="Data prefix to use for multiple runs. Defaults to test",
                        )
    parser.add_argument('--balanced',
                        help="Balance the training data to half positive, half negative.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augmented',
                        help="Augment the training data.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-flip',
                        help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-offset',
                        help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-scale',
                        help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-rotate',
                        help="Augment the training data by randomly rotating the data around the head-foot axis.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('--augment-noise',
                        help="Augment the training data by randomly adding noise to the data.",
                        action='store_true',
                        default=False,
                        )
    parser.add_argument('-c', '--config',
                        help="Use config file to overwrite defaults.",
                        default = None, type = str,
                        )
    parser.add_argument('-d', '--data_path',
                        help="Path to data",
                        default=None, type=str,
                        )
    parser.add_argument('-o', '--output_path',
                        help="Path to output directory",
                        default=None, type=str,
                        )
    parser.add_argument('-l', '--label_path',
                        help="Path to labels",
                        default=None, type=str,
                        )
    parser.add_argument('comment',
                        help="Comment suffix for Tensorboard run.",
                        nargs='?',
                        default='',
                        )

    cli_args = parser.parse_args(sys_argv)

    # update flags with config dict
    config = json2dict(cli_args.config)
    cli_args.__dict__.update(config)

    return cli_args


def json2dict(json_path: Path) -> dict:
    with open(json_path, 'rt') as json_file:
        json_config = json.load(json_file)
    return json_config


if __name__ == '__main__':
    parseConfig()
