import argparse
from modun.file_io import json2dict
import sys
from datetime import datetime


def parse_config(sys_argv=None):
    if sys_argv is None:
        sys_argv = sys.argv[1:]

    parser = argparse.ArgumentParser()

    # TRAINING
    parser.add_argument('--batch-size',
                        help='Batch size to use for training',
                        default=2,
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
    parser.add_argument('--use-augmentation',
                        help="Use data augmentation on training",
                        default=1,
                        type=int
                        )
    parser.add_argument('--weight_decay_coefficient',
                        help="Coefficient for weight decay",
                        default=1e-4,
                        type=int
                        )

    # IMAGE PARAMETERS
    parser.add_argument('--channels',
                        nargs='+', type=int,
                        default=[0, 1, 2, 3],
                        help="Index of channels to use, defaults to [0,1,2,3]",
                        )

    # VARIABLE PARAMETERS
    parser.add_argument('-op', '--outcome',
                        help="Predicted parameter",
                        default=None, type=str,
                        )

    parser.add_argument('--continuous_outcome',
                        help="Predicted parameter is continuous",
                        default=None, type=str,
                        )

    parser.add_argument('--id_variable',
                        help="ID parameter in database",
                        default=None, type=str,
                        )

    # MODEL PARAMETERS
    parser.add_argument('-mis', '--model_input_shape',
                        help="Desired input shape for model",
                        default=(46, 46, 46), type=tuple,
                        )

    # PATHS
    parser.add_argument('-c', '--config',
                        help="Use config file to overwrite defaults.",
                        default=None, type=str,
                        )
    parser.add_argument('-d', '--imaging-dataset-path',
                        help="Path to data",
                        default=None, type=str,
                        )
    parser.add_argument('-o', '--output-dir',
                        help="Path to output directory",
                        default=None, type=str,
                        )
    parser.add_argument('-l', '--label_file_path',
                        help="Path to labels",
                        default=None, type=str,
                        )

    # CROSSVALIDATION
    parser.add_argument('--cv_n_repeats',
                        help='Number of repeats for cross-validation',
                        default=1, type=int,
                        )
    parser.add_argument('--cv_n_folds',
                        help='Number of folds for cross-validation',
                        default=5, type=int,
                        )

    # MISC
    parser.add_argument('--norby', type=bool, default=False,
                        help="If true use norby package to send training updates via telegram.")

    parser.add_argument('--comment', type=str, default='',
                        help="Add comment to experiment_id")

    cli_args = parser.parse_args(sys_argv)

    # update flags with config dict
    config = json2dict(cli_args.config)
    cli_args.__dict__.update(config)
    cli_args.__dict__.update({"experiment_id": f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{cli_args.outcome}'})

    return cli_args


if __name__ == '__main__':
    parse_config()
