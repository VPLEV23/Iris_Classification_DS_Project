"""
This script prepares the data, runs the training, and saves the model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import json
import os
import sys
import mlflow


# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import get_project_dir, configure_logging

# Load configuration settings from JSON
CONF_FILE = '../settings.json'
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])


# Neural network model for the Iris dataset
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

# Function to create dataloaders


def create_dataloaders(df, batch_size=32):
    # Preprocessing
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.tolist())
            all_targets.extend(target.tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    return accuracy, precision, recall, f1
# Modified Training class for PyTorch model


class Training:
    def __init__(self):
        self.model = IrisNet()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def run_training(self, train_loader, test_loader):
        logging.info("Running training with PyTorch...")
        for epoch in range(conf['train']['epochs']):
            for data, target in train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            # Log epoch loss to MLflow
            mlflow.log_metric('loss', loss.item(), step=epoch)

            accuracy, precision, recall, f1 = evaluate_model(
                self.model, test_loader)
            mlflow.log_metric('accuracy', accuracy, step=epoch)
            mlflow.log_metric('precision', precision, step=epoch)
            mlflow.log_metric('recall', recall, step=epoch)
            mlflow.log_metric('f1_score', f1, step=epoch)
            logging.info(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {
                         accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
        logging.info("Training completed.")

    def save_model(self, model_path):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        torch.save(self.model.state_dict(), model_path)
        logging.info(f"Model saved to {model_path}")


def main():
    configure_logging()
    mlflow.autolog()

    # Data preparation
    df = pd.read_csv(TRAIN_PATH)
    train_loader, test_loader = create_dataloaders(
        df, batch_size=conf['train']['batch_size'])

    # Training
    trainer = Training()
    trainer.run_training(train_loader, test_loader)

    # Save the model
    model_save_path = os.path.join(MODEL_DIR, 'iris_pytorch_model.pth')
    trainer.save_model(model_save_path)


if __name__ == "__main__":
    main()
