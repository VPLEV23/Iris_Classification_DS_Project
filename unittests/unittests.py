import unittest
import pandas as pd
import torch
import os
import sys
import json

# Import  PyTorch model, data processing, and training classes

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from inference.run import IrisNet
from training.train import Training, create_dataloaders


CONF_FILE = 'settings.json'
MODEL_PATH = '../models/iris_pytorch_model.pth'


class TestTraining(unittest.TestCase):
    def test_create_dataloaders(self):
        df = pd.DataFrame({
            'feature1': [5.1, 4.9, 4.7, 4.6],
            'feature2': [3.5, 3.0, 3.2, 3.1],
            'feature3': [1.4, 1.4, 1.3, 1.5],
            'feature4': [0.2, 0.2, 0.2, 0.2],
            'target': [0, 0, 0, 0]
        })
        train_loader, _ = create_dataloaders(df, batch_size=2)
        for data, target in train_loader:
            self.assertEqual(data.shape[1], 4)  # 4 features
            self.assertLessEqual(len(target), 2)


    def test_train_model(self):
        tr = Training()

        df = pd.DataFrame({
            'feature1': [5.1, 4.9, 4.7, 4.6],
            'feature2': [3.5, 3.0, 3.2, 3.1],
            'feature3': [1.4, 1.4, 1.3, 1.5],
            'feature4': [0.2, 0.2, 0.2, 0.2],
            'target': [0, 0, 0, 0]
        })
        train_loader, _ = create_dataloaders(df, batch_size=2)
        tr.run_training(train_loader, _)  # Assuming run_training accepts two arguments
        self.assertIsInstance(tr.model, torch.nn.Module)

class TestInference(unittest.TestCase):
    def test_model_loading(self):
        model = IrisNet()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        self.assertIsInstance(model, IrisNet)

    def test_model_inference(self):
        model = IrisNet()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        test_input = torch.randn(1, 4)  # Random input
        with torch.no_grad():
            output = model(test_input)
        self.assertEqual(output.shape[0], 1)

if __name__ == '__main__':
    unittest.main()
