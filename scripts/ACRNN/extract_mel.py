import sys
import argparse
import pickle

from speech_utils.ACRNN.data_utils import calc_zscore, extract_mel


def main(args):
    # Map emotions with integer labels
    emot_map = {"ang": 0, "sad": 1, "hap": 2, "exc": 2, "neu": 3}
    # Calculate z-score
    print("Calculating z-scores for training data...")
    zscores = calc_zscore(
        dataset_dir=args.data_dir, num_filters=args.num_filters,
        emotions=emot_map.keys(), sessions=args.train_sess, save_path=None)
    # Extract mel filters
    print("Extracting mel filters...")
    data = extract_mel(
        dataset_dir=args.data_dir, num_filters=args.num_filters,
        emot_map=emot_map, metadata=zscores, train_sessions=args.train_sess,
        test_sessions=args.test_sess, save_path=args.save_path, eps=1e-5)
    print("Done. Data extracted is saved to", args.save_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'data_dir', type=str,
        help='Path to the `IEMOCAP_full_release` directory.')
    parser.add_argument(
        'save_path', type=str, help='Path to save the extracted data.')

    parser.add_argument(
        '--num_filters', type=int, default=40, help='Number of mel filters.')
    parser.add_argument(
        '--train_sess', type=str, nargs="+",
        default=["Session1", "Session2", "Session3", "Session4"],
        help='IEMOCAP sessions to use as training data. Passing multiple '
        'sessions is done by separating values by space.')
    parser.add_argument(
        '--test_sess', type=str, nargs="+", default=["Session5"],
        help='IEMOCAP sessions to used as validation and test data. Note that '
        'recordings where the female actor is wearing the markers are used as '
        'validation data (e.g., `Ses01F_impro01`, where F indicates "female"),'
        ' while recordings where the male actor is wearing the markers are'
        ' used as test data (e.g., `Ses01M_impro01`, where M indicates '
        '"male"). Passing multiple sessions is done by separating values by '
        'space.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
