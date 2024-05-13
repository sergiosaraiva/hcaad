# Hybrid Clustering Auto-encoder Anomaly Detection (HCAAD) for Financial Consolidation Data

## Introduction
The HCAAD program leverages a sophisticated approach combining hybrid clustering and auto-encoders to detect anomalies in financial data. This tool is invaluable for financial analysts and data scientists aiming to ensure the accuracy and integrity of consolidated financial statements and reports by identifying unusual patterns indicative of errors, fraud, or other discrepancies.

## Features
- **Dual Mode Operation:** Supports both training and predicting modes for flexible use.
- **Advanced Preprocessing and Dimension Reduction:** Implements comprehensive preprocessing and optional dimension reduction to prepare data for analysis.
- **Hybrid Clustering with HDBSCAN and DBSCAN:** Utilizes advanced clustering algorithms to detect complex patterns in data.
- **Deep Learning with Auto-encoders:** Employs auto-encoders for effective anomaly detection based on reconstruction errors.
- **Dynamic Anomaly Detection:** Dynamically determines anomaly thresholds to adapt to evolving data characteristics.

## Installation
### Prerequisites
- Python 3.8 or higher
- pip for Python package management

### Dependencies
Install all required Python libraries with pip:
```bash
pip install numpy pandas tensorflow scikit-learn hdbscan joblib
```

### Installing HCAAD
Clone the repository and install the package:
```bash
git clone https://github.com/your-username/HCAAD.git
cd HCAAD
pip install .
```

## Usage
HCAAD can be run in two modes:
1. **Training Mode**:
   ```bash
   python main.py train data_train.csv data_train
   ```
2. **Prediction Mode**:
   ```bash
   python main.py predict data_predict.csv data_clustering
   ```

### Configuration
Edit the `config.json` file to customize various parameters such as the choice of scaler, clustering algorithm, and autoencoder configurations. A sample configuration is provided in the repository.

## Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request for review.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
