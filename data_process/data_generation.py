# Importing required libraries


import json
import sys
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd


# Now you can import utils
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = '../settings.json'

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])


@singleton
class IrisDatasetHandler():
    def __init__(self):
        self.df = None

    def download_and_split(self, test_size=0.2):
        logger.info("Downloading Iris dataset...")
        iris = datasets.load_iris()
        self.df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.df['target'] = iris.target
        return train_test_split(self.df, test_size=test_size, random_state=42)

    def save(self, df: pd.DataFrame, path: str):
        logger.info(f"Saving data to {path}...")
        df.to_csv(path, index=False)


# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    iris_handler = IrisDatasetHandler()
    train_df, inference_df = iris_handler.download_and_split(
        test_size=conf['train']['test_size'])
    iris_handler.save(train_df, TRAIN_PATH)
    iris_handler.save(inference_df, INFERENCE_PATH)
    logger.info("Script completed successfully.")
