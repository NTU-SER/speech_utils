import os
import sys
import argparse

import joblib

from speech_utils.IAAN.data_utils import average_pool


def main(args):
    # Load normalized features
    features = joblib.load(args.input_path)
    # Average pooling
    features_pooled = average_pool(
        features=features, num_processes=args.num_processes,
        pool_size=args.pool_size, step_size=args.step_size,
        overlap=args.overlap_size, pad=args.pad, max_len=args.max_len)
    # Save pooled features
    joblib.dump(features_pooled, args.output_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'input_path', type=str, help='Path to the input pickle file.')
    parser.add_argument(
        'output_path', type=str,
        help='Path where the output pickle file will be saved.')

    parser.add_argument(
        'step_size', type=int, help='Step size (width of pool).')
    parser.add_argument(
        'overlap_size', type=int,
        help='Size of overlapping (must be less than step size).')

    parser.add_argument(
        '--pad', default=False, action="store_true",
        help='Whether to pad zeros to the end of the sequence.')
    parser.add_argument(
        '--max_len', type=int, default=0, help='Maximum sequence length.')
    parser.add_argument(
        '--num_processes', type=int, default=32,
        help='Number of threads in the thread pool.')
    parser.add_argument(
        '--pool_size', type=int, default=32, help='Thread pool size.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
