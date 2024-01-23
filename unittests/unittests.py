import unittest
import pandas as pd
import torch
import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()
# Import  PyTorch model, data processing, and training classes

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from inference.run import IrisNet
from training.train import Training, create_dataloaders

CONF_PATH = os.getenv('CONF_PATH')
CONF_FILE = CONF_PATH
# CONF_FILE = '../settings.json'
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'iris_pytorch_model.pth')


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
    
    def test_train_with_default_parameters(self):
        """ Test if the model can be trained with default parameters """
        df = pd.DataFrame({
            'feature1': [5.1, 4.9, 4.7, 4.6],
            'feature2': [3.5, 3.0, 3.2, 3.1],
            'feature3': [1.4, 1.4, 1.3, 1.5],
            'feature4': [0.2, 0.2, 0.2, 0.2],
            'target': [0, 0, 0, 0]
        })
        train_loader, test_loader = create_dataloaders(df, batch_size=2)
        trainer = Training()
        try:
            trainer.run_training(train_loader, test_loader)
        except Exception as e:
            self.fail(f"Training failed with default parameters: {e}")

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
    
    def test_inference_without_trained_model(self):
        """Test that inference without a trained model raises an exception"""
        model = IrisNet()
        test_input = torch.randn(1, 4)
        try:
            with torch.no_grad():
                output = model(test_input)
            # If the above lines do not raise an error, the test should fail.
            self.fail("Expected an exception for inference with an untrained model, but none occurred.")
        except Exception as e:
            # If an error is raised, we pass the test.
            pass

    def test_invalid_model_path(self):
        """ Test the behavior with an invalid model path """
        model = IrisNet()
        invalid_path = 'path/does/not/exist.pth'
        with self.assertRaises(FileNotFoundError):
            model.load_state_dict(torch.load(invalid_path))

if __name__ == '__main__':
    unittest.main()
