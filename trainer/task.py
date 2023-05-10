import argparse

from trainer import experiment


def get_args():
    """Define the task arguments with the default values.
    Returns:
        experiment parameters
    """
    args_parser = argparse.ArgumentParser()


    # Experiment arguments
    args_parser.add_argument(
        '--batch-size',
        help='Batch size for each training and evaluation step.',
        type=int,
        default=16)
    args_parser.add_argument(
        '--num-epochs',
        help="""\
        Maximum number of training data epochs on which to train.
        If both --train-size and --num-epochs are specified,
        --train-steps will be: (train-size/train-batch-size) * num-epochs.\
        """,
        default=10,
        type=int,
    )

    # Estimator arguments
    args_parser.add_argument(
        '--learning-rate',
        help='Learning rate value for the optimizers.',
        default=1e-3,
        type=float)

    # Saved model arguments
    args_parser.add_argument(
        '--bucket-name',
        default='my-models-1234',
        help='The name of the bucket where to save the model in')
    args_parser.add_argument(
        '--model-name',
        default='model',
        help='The name of your saved model')

    return args_parser.parse_args()


def main():
    """Setup / Start the experiment
    """
    args = get_args()
    print(args)
    experiment.run(args)


if __name__ == '__main__':
    main()