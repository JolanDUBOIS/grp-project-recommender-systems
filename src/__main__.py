import logging
logger = logging.getLogger(__name__)

import argparse

from src.data_normalization import data_normalization
import src.evaluation.main as evaluation_module


def setup_args():
    """ Set up command-line arguments for the main script. """
    parser = argparse.ArgumentParser(description="Recommender System Group Project")
    
    parser.add_argument("-dn", "--data-normalization", action="store_true", help="Normalize the raw data and save it to the 'data_normalized' subfolder.")
    parser.add_argument("-val", "--validation", action="store_true", help="Normalize the validation data instead of the training data.")
    parser.add_argument("-t", "--test", action="store_true", help="Run dev tests.")
    # Add more arguments as needed here
    
    return parser.parse_args()

if __name__ == "__main__":    
    args = setup_args()

    if args.data_normalization:
        data, embeddings = data_normalization(validation=args.validation, try_load=False, save=True)
    
    if args.test:
        logger.info("Running data normalization...")
        data, embeddings = data_normalization(validation=False, try_load=True)
        logger.info("Running validation")
        ##evaluation_module.validation_set_workflow(model_type="itemitem")
        evaluation_module.sliding_window_workflow(data, embeddings, model_type="baseline", TIME_WINDOW=86400)
