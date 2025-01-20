import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, losses, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import datetime
import time
import os
import json
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore")

# --- 1. Data Acquisition and Preprocessing ---
class DataHandler:
    def __init__(self, data_path='air_compressor_data.csv', simulation_size=10000, sequence_length=100, features=['pressure', 'temperature', 'flow_rate', 'vibration', 'power_consumption'], target='pressure'):
        self.data_path = data_path
        self.simulation_size = simulation_size
        self.sequence_length = sequence_length
        self.features = features
        self.target = target
        self.df = None
        self.scaler = {}
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None

    def _generate_synthetic_data(self):
        logging.info("Generating synthetic dataset...")
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=self.simulation_size, freq='H')
        data = {
            'timestamp': dates,
            'pressure': np.random.uniform(6, 10, self.simulation_size) + 0.5 * np.sin(np.linspace(0, 10 * np.pi, self.simulation_size)),
            'temperature': np.random.uniform(25, 45, self.simulation_size) + 0.2 * np.cos(np.linspace(0, 10 * np.pi, self.simulation_size)),
            'flow_rate': np.random.uniform(100, 200, self.simulation_size) + 0.3 * np.sin(np.linspace(0, 20 * np.pi, self.simulation_size)),
            'vibration': np.random.uniform(0.1, 0.5, self.simulation_size) + 0.1 * np.cos(np.linspace(0, 15 * np.pi, self.simulation_size)),
            'power_consumption': np.random.uniform(50, 100, self.simulation_size) + 0.4 * np.sin(np.linspace(0, 25 * np.pi, self.simulation_size))
        }
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def load_data(self, load_real_data=False):
        if load_real_data and os.path.exists(self.data_path):
            logging.info(f"Loading real data from: {self.data_path}")
            self.df = pd.read_csv(self.data_path, index_col='timestamp', parse_dates=True)
        else:
            logging.info("Using synthetic data.")
            self.df = self._generate_synthetic_data()
        logging.info(f"Data loaded with shape: {self.df.shape}")
        return self.df

    def data_exploration(self):
        if self.df is None:
            logging.error("Data not loaded. Please load data first.")
            return
        logging.info("Performing data exploration...")
        print(self.df.head())
        print(self.df.info())
        print(self.df.describe())
        self._plot_time_series()

    def _plot_time_series(self):
        num_features = len(self.features)
        fig, axes = plt.subplots(num_features, 1, figsize=(15, 5 * num_features))
        if num_features == 1:
            axes = [axes]
        for i, feature in enumerate(self.features):
            axes[i].plot(self.df[feature])
            axes[i].set_title(f'{feature} Time Series')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(feature)
        plt.tight_layout()
        plt.show()

    def preprocess_data(self):
        if self.df is None:
            logging.error("Data not loaded. Please load data first.")
            return

        logging.info("Preprocessing data...")
        # Handle missing values
        self.df.fillna(method='ffill', inplace=True) # Forward fill
        self.df.fillna(method='bfill', inplace=True) # Backward fill if still NaN

        # Scale data
        for feature in self.features + [self.target]:
          scaler = MinMaxScaler()
          self.df[feature] = scaler.fit_transform(self.df[[feature]])
          self.scaler[feature] = scaler # Store the scalers
        return self.df

    def create_sequences(self, data, sequence_length):
        sequences = []
        targets = []
        for i in range(len(data) - sequence_length):
            seq = data.iloc[i:i + sequence_length][self.features].values
            target = data.iloc[i + sequence_length][self.target]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    def split_data(self, test_size=0.2, val_size=0.1):
        if self.df is None:
            logging.error("Data not loaded. Please load data first.")
            return

        logging.info("Splitting data into train, validation, and test sets...")

        X, y = self.create_sequences(self.df, self.sequence_length)

        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(1-test_size), shuffle=False) #Adjust validation size

        logging.info(f"Train set size: {self.X_train.shape}")
        logging.info(f"Validation set size: {self.X_val.shape}")
        logging.info(f"Test set size: {self.X_test.shape}")


    def get_data(self):
       if self.X_train is None:
          logging.error("Data not split. Please call split_data() first.")
          return None, None, None, None, None, None
       return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

    def inverse_transform(self, scaled_predictions, feature_name):
      if not self.scaler or feature_name not in self.scaler:
        logging.error(f"Scaler for {feature_name} not found.")
        return scaled_predictions
      scaler = self.scaler[feature_name]
      dummy_data = np.zeros((len(scaled_predictions), len(self.features)))
      dummy_data[:, self.features.index(feature_name)] = scaled_predictions.flatten()
      return scaler.inverse_transform(dummy_data)[:, self.features.index(feature_name)]

# --- 2. Model Architecture Design ---
class ModelBuilder:
    def __init__(self, sequence_length, num_features, model_type='lstm', lstm_units=50, cnn_filters=32, cnn_kernel_size=3, attention=False, hybrid=False, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model_type = model_type
        self.lstm_units = lstm_units
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.attention = attention
        self.hybrid = hybrid
        self.dropout_rate = dropout_rate

    def build_lstm(self):
        logging.info("Building LSTM model...")
        model = models.Sequential()
        model.add(layers.LSTM(self.lstm_units, activation='relu', input_shape=(self.sequence_length, self.num_features), return_sequences=False))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(1))
        return model

    def build_cnn(self):
        logging.info("Building CNN model...")
        model = models.Sequential()
        model.add(layers.Conv1D(filters=self.cnn_filters, kernel_size=self.cnn_kernel_size, activation='relu', input_shape=(self.sequence_length, self.num_features)))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(self.lstm_units, activation='relu')) #Dense Layer
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(1))
        return model


    def build_cnn_lstm(self):
        logging.info("Building CNN-LSTM hybrid model...")
        model = models.Sequential()
        model.add(layers.Conv1D(filters=self.cnn_filters, kernel_size=self.cnn_kernel_size, activation='relu', input_shape=(self.sequence_length, self.num_features)))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.LSTM(self.lstm_units, activation='relu', return_sequences=False))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(1))
        return model


    def build_attention_lstm(self):
        logging.info("Building LSTM with attention model...")
        input_layer = layers.Input(shape=(self.sequence_length, self.num_features))
        lstm_out, _, _ = layers.LSTM(self.lstm_units, return_sequences=True, return_state=True)(input_layer)
        attention_weights = layers.Dense(1, activation='tanh')(lstm_out) #Attention Weight
        attention_weights = layers.Flatten()(attention_weights)
        attention_weights = layers.Activation('softmax')(attention_weights)
        attention_weights = layers.RepeatVector(self.lstm_units)(attention_weights)
        attention_weights = layers.Permute([2, 1])(attention_weights) #Permute for matrix multiplication
        attention_applied = layers.multiply([lstm_out, attention_weights])
        attention_applied = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attention_applied) #Sum along time axis
        out = layers.Dense(1)(attention_applied)
        model = models.Model(inputs=input_layer, outputs=out)
        return model

    def build_model(self):
        if self.model_type == 'lstm':
            return self.build_lstm()
        elif self.model_type == 'cnn':
            return self.build_cnn()
        elif self.model_type == 'cnn_lstm' or self.hybrid:
            return self.build_cnn_lstm()
        elif self.model_type == 'attention_lstm' or self.attention:
            return self.build_attention_lstm()
        else:
           logging.error(f"Invalid model type: {self.model_type}")
           return None

# --- 3. Model Training and Evaluation ---
class ModelTrainer:
    def __init__(self, model, learning_rate=0.001, epochs=100, batch_size=32, patience=10, log_dir='logs', model_path='best_model.h5'):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.log_dir = log_dir
        self.model_path = model_path
        self.history = None


    def compile_model(self):
        logging.info("Compiling the model...")
        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss=losses.MeanSquaredError(), metrics=[metrics.MeanAbsoluteError()])

    def train_model(self, X_train, y_train, X_val, y_val):
      if self.model is None:
        logging.error("Model not initialized. Build model first.")
        return

      self.compile_model()

      # Create tensorboard logging callback
      tensorboard_callback = callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
      # Create checkpoint callback to save the best model
      checkpoint_callback = callbacks.ModelCheckpoint(filepath=self.model_path, monitor='val_loss', save_best_only=True, save_weights_only=False)
      # Create early stopping callback to stop training early
      early_stopping_callback = callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

      logging.info("Starting training...")
      start_time = time.time()
      self.history = self.model.fit(X_train, y_train,
                                   validation_data=(X_val, y_val),
                                   epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback])
      end_time = time.time()
      training_time = end_time - start_time
      logging.info(f"Training completed in {training_time:.2f} seconds.")

    def evaluate_model(self, X_test, y_test):
      if self.model is None:
        logging.error("Model not initialized. Build and train model first.")
        return

      logging.info("Evaluating model...")
      if os.path.exists(self.model_path):
        logging.info(f"Loading saved weights from: {self.model_path}")
        self.model.load_weights(self.model_path)
      loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
      logging.info(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")
      return loss, mae

    def predict(self, X):
      if self.model is None:
        logging.error("Model not initialized. Build and train model first.")
        return
      if os.path.exists(self.model_path):
          logging.info(f"Loading model from: {self.model_path}")
          self.model = models.load_model(self.model_path)
      predictions = self.model.predict(X)
      return predictions

    def plot_training_history(self):
        if self.history is None:
            logging.error("Model not trained. Please train the model first.")
            return

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mean_absolute_error'], label='Training MAE')
        plt.plot(self.history.history['val_mean_absolute_error'], label='Validation MAE')
        plt.title('MAE Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_predictions(self, y_true, y_pred, feature_name):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'Actual vs Predicted {feature_name}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def calculate_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        logging.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return rmse, mae

    def save_history(self, file_path):
        if self.history is not None:
            logging.info(f"Saving training history to: {file_path}")
            history_data = self.history.history
            with open(file_path, 'w') as f:
              json.dump(history_data, f)
        else:
            logging.warning("No training history to save.")

    def load_history(self, file_path):
        if os.path.exists(file_path):
            logging.info(f"Loading training history from: {file_path}")
            with open(file_path, 'r') as f:
                history_data = json.load(f)
            self.history = type('history_object', (object,), {'history':history_data})()

        else:
          logging.warning("No training history to load.")

# --- 4. Advanced Techniques and Experimentation ---
class ExperimentManager:
    def __init__(self, data_handler, base_config, experiment_dir='experiments'):
        self.data_handler = data_handler
        self.base_config = base_config
        self.experiment_dir = experiment_dir
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.results = []

    def run_experiment(self, exp_config, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        exp_name = self._generate_exp_name(exp_config)
        exp_dir = os.path.join(self.experiment_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        log_path = os.path.join(exp_dir, 'experiment_log.txt')
        model_path = os.path.join(exp_dir, 'best_model.h5')
        history_path = os.path.join(exp_dir, 'history.json')


        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(file_handler)


        logging.info(f"Starting experiment: {exp_name}")
        logging.info(f"Experiment configuration: {exp_config}")


        sequence_length = exp_config.get('sequence_length', self.base_config['sequence_length'])
        num_features = len(self.data_handler.features)
        model_type = exp_config.get('model_type', self.base_config['model_type'])
        lstm_units = exp_config.get('lstm_units', self.base_config['lstm_units'])
        cnn_filters = exp_config.get('cnn_filters', self.base_config['cnn_filters'])
        cnn_kernel_size = exp_config.get('cnn_kernel_size', self.base_config['cnn_kernel_size'])
        attention = exp_config.get('attention', self.base_config['attention'])
        hybrid = exp_config.get('hybrid', self.base_config['hybrid'])
        dropout_rate = exp_config.get('dropout_rate', self.base_config['dropout_rate'])
        learning_rate = exp_config.get('learning_rate', self.base_config['learning_rate'])
        epochs = exp_config.get('epochs', self.base_config['epochs'])
        batch_size = exp_config.get('batch_size', self.base_config['batch_size'])
        patience = exp_config.get('patience', self.base_config['patience'])

        model_builder = ModelBuilder(sequence_length, num_features, model_type, lstm_units, cnn_filters, cnn_kernel_size, attention, hybrid, dropout_rate)
        model = model_builder.build_model()
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_handler.get_data()

        model_trainer = ModelTrainer(model, learning_rate, epochs, batch_size, patience, log_dir=os.path.join(exp_dir,'logs'), model_path=model_path)
        model_trainer.train_model(X_train, y_train, X_val, y_val)
        test_loss, test_mae = model_trainer.evaluate_model(X_test, y_test)

        test_predictions = model_trainer.predict(X_test)
        test_predictions_original = self.data_handler.inverse_transform(test_predictions, self.data_handler.target)
        y_test_original = self.data_handler.inverse_transform(y_test, self.data_handler.target)
        rmse, mae = model_trainer.calculate_metrics(y_test_original, test_predictions_original)

        model_trainer.plot_training_history()
        model_trainer.plot_predictions(y_test_original, test_predictions_original, self.data_handler.target)
        model_trainer.save_history(history_path)


        result = {
            'experiment_name': exp_name,
            'config': exp_config,
            'test_loss': test_loss,
            'test_mae': test_mae,
             'test_rmse': rmse,
            'test_predictions': test_predictions_original.tolist(),
             'true_values': y_test_original.tolist()
        }
        self.results.append(result)
        logging.info(f"Experiment completed. Results: {result}")
        logger.removeHandler(file_handler) # remove handler after experiment

    def _generate_exp_name(self, exp_config):
        model_str = exp_config.get('model_type', 'default')
        lstm_str = f'lstm{exp_config.get("lstm_units","")}' if exp_config.get('lstm_units') else ""
        cnn_str = f'cnn{exp_config.get("cnn_filters","")}' if exp_config.get('cnn_filters') else ""
        attn_str = "attn" if exp_config.get('attention') else ""
        hybrid_str = "hybrid" if exp_config.get('hybrid') else ""
        lr_str = f'lr{exp_config.get("learning_rate",self.base_config["learning_rate"])}'
        return f'exp_{model_str}{lstm_str}{cnn_str}{attn_str}{hybrid_str}_{lr_str}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'

    def save_results(self, filename='all_experiment_results.json'):
       logging.info(f"Saving all experiment results to: {filename}")
       with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)

    def load_results(self, filename='all_experiment_results.json'):
       if os.path.exists(filename):
           logging.info(f"Loading all experiment results from: {filename}")
           with open(filename, 'r') as f:
               self.results = json.load(f)
       else:
           logging.warning("No experiment results file found.")

# --- 5. Deployment ---
class Deployment:
    def __init__(self, model_path, data_handler):
        self.model_path = model_path
        self.model = None
        self.data_handler = data_handler

    def load_model(self):
        if os.path.exists(self.model_path):
          logging.info(f"Loading the model from {self.model_path}...")
          self.model = models.load_model(self.model_path)
        else:
            logging.error("Model path does not exist")

    def batch_predict(self, X):
      if self.model is None:
         logging.error("Model not loaded. Please load model first.")
         return None

      logging.info("Performing batch prediction...")
      predictions = self.model.predict(X)
      predictions = self.data_handler.inverse_transform(predictions, self.data_handler.target)
      return predictions

    def simple_api(self, data):
      if self.model is None:
        logging.error("Model not loaded. Please load model first.")
        return None
      
      logging.info("Performing prediction using api...")
      data = np.array(data).reshape(1, self.data_handler.sequence_length, len(self.data_handler.features)) #Reshape for one prediction
      prediction = self.model.predict(data)
      prediction = self.data_handler.inverse_transform(prediction, self.data_handler.target)
      return prediction.flatten()[0]


# --- 6. Research Documentation and Reporting ---
class ResearchReport:
    def __init__(self, experiment_manager, report_path='report.txt'):
        self.experiment_manager = experiment_manager
        self.report_path = report_path
        self.report_content = ""
    def generate_report(self):
      logging.info("Generating research report...")
      self.report_content += "Air Compressor Predictive Analysis Research Report\n"
      self.report_content += "============================================\n\n"

      self.report_content += "Experiment Overview:\n"
      self.report_content += f"Total Experiments: {len(self.experiment_manager.results)}\n\n"
      if not self.experiment_manager.results:
         self.report_content += "No experiments results found. Please run experiments first.\n"
         self._save_report()
         return


      self.report_content += "Experiment Results:\n"
      for result in self.experiment_manager.results:
           self.report_content += f"  Experiment: {result['experiment_name']}\n"
           self.report_content += f"  Config: {result['config']}\n"
           self.report_content += f"  Test Loss: {result['test_loss']:.4f}\n"
           self.report_content += f"  Test MAE: {result['test_mae']:.4f}\n"
           self.report_content += f"  Test RMSE: {result['test_rmse']:.4f}\n"
           self.report_content += "  ------------------------------------------\n"

      self.report_content += "\nTop Performing Experiments:\n"
      sorted_results = sorted(self.experiment_manager.results, key=lambda x: x['test_mae'])
      for i, result in enumerate(sorted_results[:3]):
           self.report_content += f"  {i+1}. Experiment: {result['experiment_name']}\n"
           self.report_content += f"     Config: {result['config']}\n"
           self.report_content += f"     Test MAE: {result['test_mae']:.4f}\n"
           self.report_content += f"     Test RMSE: {result['test_rmse']:.4f}\n"

      self.report_content += "\nOverall Summary:\n"
      min_mae_exp = min(self.experiment_manager.results, key=lambda x: x['test_mae'])
      self.report_content += f"  Best Experiment (Lowest MAE): {min_mae_exp['experiment_name']}\n"
      self.report_content += f"  Best MAE: {min_mae_exp['test_mae']:.4f}\n"

      self._save_report()

    def _save_report(self):
        logging.info(f"Saving research report to: {self.report_path}")
        with open(self.report_path, 'w') as f:
            f.write(self.report_content)

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize Data Handler
    data_handler = DataHandler(simulation_size=10000, sequence_length=100)  #Adjust sequence_length, features as needed
    data = data_handler.load_data(load_real_data=False)  # Set to true if using real data
    data_handler.data_exploration()
    data_handler.preprocess_data()
    data_handler.split_data(test_size=0.2, val_size=0.1)

    # Experiment Configuration
    base_config = {
        'sequence_length': 100,
        'model_type': 'lstm',
        'lstm_units': 50,
        'cnn_filters': 32,
        'cnn_kernel_size': 3,
        'attention': False,
        'hybrid': False,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'patience': 10
    }
    experiment_manager = ExperimentManager(data_handler, base_config)

    # Define Experiments to run
    experiments = [
        {'model_type': 'lstm', 'lstm_units': 100, 'epochs': 10},
        {'model_type': 'cnn','cnn_filters': 64, 'epochs': 10},
        {'model_type': 'cnn_lstm', 'lstm_units': 100, 'cnn_filters': 64, 'epochs': 10},
        {'model_type': 'attention_lstm', 'lstm_units': 100, 'epochs': 10},
        {'model_type': 'lstm', 'lstm_units': 100, 'dropout_rate': 0.3, 'epochs': 10}
    ]

    # Run Experiments
    for exp_config in experiments:
        experiment_manager.run_experiment(exp_config)
    experiment_manager.save_results()

    # Generate Research Report
    research_report = ResearchReport(experiment_manager)
    research_report.generate_report()

    #Load results and generate a new report
    experiment_manager.load_results()
    research_report = ResearchReport(experiment_manager)
    research_report.generate_report()

    # Model Deployment
    best_exp = min(experiment_manager.results, key=lambda x: x['test_mae'])
    best_model_path = os.path.join("experiments", best_exp['experiment_name'], "best_model.h5")
    deployment = Deployment(best_model_path, data_handler)
    deployment.load_model()

    X_test, y_test, _, _, _, _ = data_handler.get_data()
    #Batch prediction
    batch_predictions = deployment.batch_predict(X_test)
    # Prediction using api
    sample_data = X_test[0] #Take one sample
    prediction = deployment.simple_api(sample_data)
    logging.info(f"API Prediction output: {prediction}")


    logging.info("Predictive analysis of air compressor completed")