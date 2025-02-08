import argparse

from src.data_normalization import data_normalization


def setup_args():
    """ Set up command-line arguments for the main script. """
    parser = argparse.ArgumentParser(description="Recommender System Group Project")
    
    parser.add_argument("-dn", "--data-normalization", action="store_true", help="Normalize the raw data and save it to the 'data_normalized' subfolder.")
    parser.add_argument("-val", "--validation", action="store_true", help="Normalize the validation data instead of the training data.")
    # Add more arguments as needed here
    
    return parser.parse_args()

if __name__ == "__main__":    
    args = setup_args()

    if args.data_normalization:
        data = data_normalization(validation=args.validation, try_load=False, save=True)
