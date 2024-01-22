"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""
import torch
import torch.nn as nn
import argparse
import json
import logging
import os
import sys
import pandas as pd
import torch
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = '../settings.json'

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file",
                    help="Specify inference data file",
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path",
                    help="Specify the path to the output table")


class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 50)  # 4 input features
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 3)  # 3 output classes

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pickle') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.pickle'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)


def get_model_by_path(path: str) -> IrisNet:
    """Loads and returns the specified PyTorch model"""
    model = IrisNet()  # Initialize your model
    model.load_state_dict(torch.load(path))
    model.eval()
    logging.info(f'Path of the model: {path}')
    return model


def prepare_data(df: pd.DataFrame) -> DataLoader:
    """Prepares pandas DataFrame for inference"""
    # Select only the relevant feature columns. Adjust the column names as per your data.
    feature_columns = [
        'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    df_features = df[feature_columns]

    scaler = StandardScaler()
    features = scaler.fit_transform(df_features.values)
    tensors = torch.tensor(features, dtype=torch.float32)
    dataset = TensorDataset(tensors)
    loader = DataLoader(dataset, batch_size=32)  # Adjust batch size as needed
    return loader


def predict_results(model: IrisNet, loader: DataLoader) -> pd.DataFrame:
    """Predict the results"""
    model.eval()
    results = []
    with torch.no_grad():
        for data in loader:
            outputs = model(data[0])
            _, predicted = torch.max(outputs, 1)
            results.extend(predicted.numpy())
    return results


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(
            conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    configure_logging()
    args = parser.parse_args()

    model_path = get_latest_model_path()
    model = get_model_by_path(model_path)
    infer_file = os.path.join(DATA_DIR, args.infer_file)
    infer_data = pd.read_csv(infer_file)
    loader = prepare_data(infer_data)
    results = predict_results(model, loader)
    store_results(results, args.out_path)

    logging.info(f'Prediction results stored.')


if __name__ == "__main__":
    main()
