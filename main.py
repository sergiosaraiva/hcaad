import os
import sys
import pandas as pd
import json
import logging
from hcaad import HCAAD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def load_config():
    try:
        with open('config.json', 'r') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

def load_data(file_path):
    try:
        return pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        logger.error(f"Error: File not found - {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred when loading data: {e}")
        sys.exit(1)

def train(file_path, model_path):
    config = load_config()
    data = load_data(file_path)
    hcaad = HCAAD(config)
    
    data_preprocessed = hcaad.preprocess_data(data, fit_transform=True)
    cluster_labels = hcaad.apply_clustering(data_preprocessed)
    hcaad.build_autoencoder(data_preprocessed.shape[1])
    hcaad.train_autoencoder(data_preprocessed)
    hcaad.save_model(model_path)
    logger.info("Model trained and saved to " + model_path)

def predict(file_path, model_path):
    config = load_config()
    data = load_data(file_path)
    hcaad = HCAAD(config)
    
    # Ensure the model and transformer are loaded.
    hcaad.load_model(model_path)
    
    # Transform data using the loaded transformer.
    data_preprocessed = hcaad.preprocess_data(data, fit_transform=False)
    cluster_labels = hcaad.apply_clustering(data_preprocessed)
    anomalies = hcaad.detect_anomalies(data, data_preprocessed, cluster_labels)
    
    logger.info("Detected anomalies:")
    logger.info(anomalies)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        logger.error("Usage: python main.py <command> <file_path> <model_path>")
        sys.exit(1)

    command = sys.argv[1]
    file_path = sys.argv[2]
    model_path = sys.argv[3]

    if command == 'train':
        train(file_path, model_path)
    elif command == 'predict':
        predict(file_path, model_path)
    else:
        logger.error("Invalid command. Use 'train' or 'predict'.")
        sys.exit(1)
