import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy import sparse
import hdbscan
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class HCAAD:
    def __init__(self, config):
        self.config = config
        self.column_transformer = None

    def optimize_data_types(self, df):
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype('float32')
            elif pd.api.types.is_integer_dtype(df[col]):
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype('int8')
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype('int16')
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype('int32')

    def preprocess_data(self, data, fit_transform=True):
        self.optimize_data_types(data)
        
        numeric_features = self.config['features']['numeric_features']
        categorical_features = self.config['features']['categorical_features']
        
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.column_transformer = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop')
        
        if fit_transform:
            data_preprocessed = self.column_transformer.fit_transform(data)
        else:
            data_preprocessed = self.column_transformer.transform(data)
        
        return data_preprocessed


    def apply_clustering(self, data_scaled):
        cluster_algo = self.config.get('clustering_algorithm', 'hdbscan').lower()
        if cluster_algo == 'dbscan':
            return self.dbscan_clustering(data_scaled)
        elif cluster_algo == 'hdbscan':
            if sparse.issparse(data_scaled):
                # Ensure data is in a compatible format
                data_scaled = data_scaled.toarray()
            return self.hdbscan_clustering(data_scaled)
        else:
            logger.info("Skipping clustering")
            return None

    def dbscan_clustering(self, data_scaled):
        dbscan_config = self.config.get('dbscan', {})
        dbscan = DBSCAN(eps=dbscan_config['eps'], min_samples=dbscan_config['min_samples'])
        return dbscan.fit_predict(data_scaled)

    def hdbscan_clustering(self, data_scaled):
        hdbscan_config = self.config.get('hdbscan', {})
        clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_config['min_cluster_size'], min_samples=hdbscan_config['min_samples'])
        return clusterer.fit_predict(data_scaled)

    def build_autoencoder(self, input_dim):
        layers_config = self.config['autoencoder']
        activation = layers_config['hidden_layers_activation']
        output_activation = layers_config['output_layer_activation']

        input_layer = Input(shape=(input_dim,))
        x = input_layer
        for neurons in layers_config['hidden_layers']:
            x = Dense(neurons, activation=activation)(x)
            x = Dropout(layers_config['dropout_rate'])(x)
        output_layer = Dense(input_dim, activation=output_activation)(x)

        self.autoencoder = Model(input_layer, output_layer)
        self.autoencoder.compile(optimizer=Adam(learning_rate=layers_config['learning_rate']), loss=layers_config['loss'])
        logger.info(self.autoencoder.summary())
        return self.autoencoder

    def train_autoencoder(self, data):
        # Convert scipy sparse matrix to dense
        if isinstance(data, sparse.spmatrix):
            data = data.toarray()

        # TensorFlow SparseTensor check already included, now ensuring it handles scipy sparse as well
        if tf.is_tensor(data) and isinstance(data, tf.sparse.SparseTensor):
            data = tf.sparse.to_dense(data)
        
        # Early stopping configuration
        early_stopping_config = self.config['autoencoder'].get("apply_early_stopping", False)
        callbacks = [EarlyStopping(
            monitor='val_loss',
            patience=self.config['autoencoder'].get('patience', 10),
            restore_best_weights=self.config['autoencoder'].get('restore_best_weights', True)
        )] if early_stopping_config else []

        # Model training
        self.autoencoder.fit(
            data, data,
            epochs=self.config['autoencoder']['epochs'],
            batch_size=self.config['autoencoder']['batch_size'],
            validation_split=self.config['autoencoder']['validation_split'],
            callbacks=callbacks
        )

    def detect_anomalies(self, data, data_scaled, cluster_labels):
        reconstructed = self.autoencoder.predict(data_scaled)
        reconstruction_error = np.mean(np.power(data_scaled - reconstructed, 2), axis=1)
        threshold = np.mean(reconstruction_error) + self.config['anomaly_detection']['threshold_multiplier'] * np.std(reconstruction_error)
        anomaly_mask = reconstruction_error > threshold
        if cluster_labels is not None:
            anomaly_mask |= (cluster_labels == -1)
        anomalies = data[anomaly_mask].copy()
        anomalies['ReconstructionError'] = reconstruction_error[anomaly_mask]
        return anomalies

    def save_model(self, model_path):
        self.autoencoder.save(model_path + '.keras')
        joblib.dump(self.column_transformer, model_path + '_preprocessor.pkl')

    def load_model(self, model_path):
        self.autoencoder = load_model(model_path + '.keras')
        self.column_transformer = joblib.load(model_path + '_preprocessor.pkl')
