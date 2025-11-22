# DSP Project - Network Traffic Anomaly Detection

This project implements deep learning models for network traffic anomaly detection using various neural network architectures.

## Project Structure

### Main Files

- **main.py** - Main entry point for LSTM-based anomaly detection
- **DenseNN.py** - Dense Neural Network implementation (best performing model)
- **comparer.py** - Comprehensive architecture comparison (Dense, RNN, LSTM, GRU, Bidirectional LSTM)
- **comparing_methods.py** - Advanced RNN architectures for traffic prediction
- **Attention.py** - Attention mechanism implementations and transformer encoder
- **Simple_RNN.py** - Custom RNN implementation from scratch
- **lightweight_tuning.py** - Hyperparameter tuning utilities
- **datasetcheck.py** - Dataset exploration utility

### Key Features

- Multiple neural network architectures (Dense, RNN, LSTM, GRU, Bidirectional LSTM)
- Attention mechanisms for feature importance
- Hyperparameter tuning utilities
- Comprehensive evaluation metrics
- Visualization of results

## Usage

Run the main script:
```bash
python main.py
```

Or run specific components:
```bash
python DenseNN.py
python comparer.py
```

## Requirements

- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

