# Iris Classification Project with Docker and PyTorch

Welcome to the Iris Classification Project. This project showcases a complete workflow for training a deep learning model using PyTorch on the Iris dataset, containerizing the training and inference processes with Docker, and ensuring code quality with comprehensive unit tests.

## Installation

To set up this project, clone the repository and install the required dependencies:

```
git clone https://github.com/VPLEV23/Iris_Classification_DS_Project
cd iris_classification_ds_project
```
### Setting Up a Local Development Environment

It is recommended to create a virtual environment for development to manage dependencies effectively:

1. **Create a Virtual Environment** (optional but recommended):
   
   If you're using Python 3, a virtual environment can be created by running:

   ```
   python -m venv env
   ```

   On Windows, you can activate the environment with:

   ```
   .\env\Scripts\activate
   ```

   On macOS and Linux, use:

   ```
   source env/bin/activate
   ```

2. **Install Required Packages**:

   After activating your environment, install the required packages by running:

   ```
   pip install -r requirements.txt
   ```

This will set up a local environment for you to develop and test the Iris Classification Project.

Make sure Docker is installed on your system. If not, you can install it from [Docker's official website](https://www.docker.com/get-started).

### Installing MLFlow on Windows

MLFlow can be easily installed on a Windows local machine using the pip, the Python package installer. To do so, open the command prompt (you can find it by searching for `cmd` in the Start menu) and type the following command:

```python
pip install mlflow
```

After the successful installation, you can start managing and deploying your ML models with MLFlow. For further information on how to use MLFlow at its best, refer to the official MLFlow documentation or use the `mlflow --help` command.

Should you encounter any issues during the installation, you can bypass them by commenting out the corresponding lines in the `train.py` and `requirements.txt` files.

To run MLFlow, type `mlflow ui` in your terminal and press enter. If it doesn't work, you may also try `python -m mlflow ui`  This will start the MLFlow tracking UI, typically running on your localhost at port 5000. You can then access the tracking UI by opening your web browser and navigating to `http://localhost:5000`.


## Usage

### Settings:
The configurations for the project are managed using the `settings.json` file. It stores important variables that control the behaviour of the project. Examples could be the path to certain resource files, constant values, hyperparameters for an ML model, or specific settings for different environments. Before running the project, ensure that all the paths and parameters in `settings.json` are correctly defined.
Keep in mind that you may need to pass the path to your config to the scripts. For this, you may create a .env file or manually initialize an environment variable as `CONF_PATH=settings.json`.
Please note, some IDEs, including VSCode, may have problems detecting environment variables defined in the .env file. This is usually due to the extension handling the .env file. If you're having problems, try to run your scripts in a debug mode, or, as a workaround, you can hardcode necessary parameters directly into your scripts. Make sure not to expose sensitive data if your code is going to be shared or public. In such cases, consider using secret management tools provided by your environment.
### Training the Model

1. Generate the dataset by running `data_generation.py`.
2. Build and run the Docker container for training:
- Build the training Docker image. If the built is successfully done, it will automatically train the model:
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
- You may run the container with the following parameters to ensure that the trained model is here:
```bash
docker run -it training_image /bin/bash
```
Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:
```bash
docker cp <container_id>:/app/models/<model_name>.pth ./models
```
Replace `<container_id>` with your running Docker container ID and `<model_name>.pth` with your model's name.

3. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python .\training\train.py
```

### Inference

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name>.pth --build-arg settings_name=settings.json -t inference_image .
```
- Run the inference Docker container:
```bash
docker run -v /path_to_your_local_model_directory:/app/models -v /path_to_your_input_folder:/app/input -v /path_to_your_output_folder:/app/output inference_image
```
- Or you may run it with the attached terminal using the following command:
```bash
docker run -it inference_image /bin/bash  
```
After that ensure that you have your results in the `results` directory in your inference container.
2. Alternatively, the `run.py` script can also be run locally as follows:

```bash
python .\inference\run.py
```

## Project Structure
```
Iris_Classification_DS_Project
├── data                      # Data files used for training and inference (it can be generated with data_generation.py script)
│   ├── iris_inference_data.csv
│   └── iris_training_data.csv
├── data_process              # Scripts used for data processing and generation
│   ├── data_generation.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── various model files
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── utils.py                  # Utility functions and classes that are used in scripts
├── settings.json             # All configurable parameters and settings
└── README.md
```

## Unit Tests

Run unit tests to ensure code reliability:

```
python .\unittests\unittests.py
```

## Contributing

Contributions to this project are welcome. Please adhere to conventional pull request protocols.


## Contact

For any inquiries, feel free to reach out at [bladetheblade12@gmail.com](mailto:bladetheblade12@gmail.com).
