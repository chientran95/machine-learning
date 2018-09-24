import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        type=str,
        default='hpt',
    )
    parser.add_argument(
        '--max_iter',
        type=int,
        default=30,
    )
    parser.add_argument(
        '--trial_steps',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--n_bags',
        type=int,
        default=3,
    )
    args, _ = parser.parse_known_args()

    return args