import os
import sys
import argparse
from shutil import rmtree

import joblib

from speech_utils.IAAN.data_utils import extract, normalize


def main(args):
    # Extract features to an intermediate folder
    print("Extracting speech features to", args.features_dir)
    if os.path.exists(args.features_dir):
        rmtree(args.features_dir)  # need to remove existing folder
    os.mkdir(args.features_dir)
    extract(iemocap_dir=args.data_dir, out_dir=args.features_dir,
            smile_path=args.smile_path, smile_conf=args.smile_conf,
            num_processes=args.num_processes, step_size=args.step_size)
    # Normalize features with respect to speakers
    print("Normalizing features")
    features_normalized = normalize(
        features_dir=args.features_dir, num_processes=args.num_processes)
    # Save normalized features
    joblib.dump(features_normalized, args.save_path)
    print("Successfully save normalized features to", args.save_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'data_dir', type=str,
        help='Path to the `IEMOCAP_full_release` directory.')
    parser.add_argument(
        'features_dir', type=str,
        help='Directory used to save intermediate feature files extracted '
             'from openSMILE (as an external executable).')
    parser.add_argument(
        'save_path', type=str, help='Path to save the normalized features.')

    parser.add_argument(
        'smile_path', type=str,
        help='Path to the openSMILE executable (i.e., SMILExtract).')
    parser.add_argument(
        '--smile_conf', type=str, default="emobase_v2.conf",
        help='Path to the openSMILE configuration.')

    parser.add_argument(
        '--num_processes', type=int, default=32,
        help='Number of threads in the thread pool.')
    parser.add_argument(
        '--step_size', type=int, default=32, help='Step size for thread pool.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
