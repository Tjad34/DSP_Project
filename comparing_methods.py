import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# ============================================================
# MODERN RNN ARCHITECTURES FOR NETWORK TRAFFIC PREDICTION
# ============================================================

def create_simple_rnn_model(input_shape, output_size, task='regression',
                            rnn1_units=64, rnn2_units=32, dense_units=16,
                            rnn_dropout=0.3, rnn_recurrent_dropout=0.3,
                            dropout1=0.4, dropout2=0.4, dropout3=0.3,
                            l2_reg=0.001):
    """
    Basic RNN model (baseline from the paper)
    Prone to vanishing gradients but useful for comparison
    
    Hyperparameters:
    - rnn1_units: Number of units in first RNN layer
    - rnn2_units: Number of units in second RNN layer
    - dense_units: Number of units in dense layer
    - rnn_dropout: Dropout rate for RNN layers
    - rnn_recurrent_dropout: Recurrent dropout rate for RNN layers
    - dropout1: Dropout after first RNN layer
    - dropout2: Dropout after second RNN layer
    - dropout3: Dropout after dense layer
    - l2_reg: L2 regularization strength
    """
    model = models.Sequential([
        layers.SimpleRNN(rnn1_units, return_sequences=True, input_shape=input_shape, 
                        dropout=rnn_dropout, recurrent_dropout=rnn_recurrent_dropout),
        layers.Dropout(dropout1),
        layers.SimpleRNN(rnn2_units, return_sequences=False, 
                        dropout=rnn_dropout, recurrent_dropout=rnn_recurrent_dropout),
        layers.Dropout(dropout2),
        layers.Dense(dense_units, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout3),
        layers.Dense(output_size, activation='linear' if task == 'regression' else 'softmax')
    ])
    return model

def create_lstm_model(input_shape, output_size, task='regression'):
    """
    Standard LSTM model with dropout for regularization
    Good for capturing long-term dependencies
    """
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape,
                   dropout=0.3, recurrent_dropout=0.3),
        layers.Dropout(0.4),
        layers.LSTM(32, return_sequences=False,
                   dropout=0.3, recurrent_dropout=0.3),
        layers.Dropout(0.4),
        layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(output_size, activation='linear' if task == 'regression' else 'softmax')
    ])
    return model

def create_gru_model(input_shape, output_size, task='regression'):
    """
    GRU model - computationally more efficient than LSTM
    Similar performance with fewer parameters
    """
    model = models.Sequential([
        layers.GRU(64, return_sequences=True, input_shape=input_shape,
                  dropout=0.3, recurrent_dropout=0.3),
        layers.Dropout(0.4),
        layers.GRU(32, return_sequences=False,
                  dropout=0.3, recurrent_dropout=0.3),
        layers.Dropout(0.4),
        layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(output_size, activation='linear' if task == 'regression' else 'softmax')
    ])
    return model

def create_bidirectional_lstm(input_shape, output_size, task='regression'):
    """
    Bidirectional LSTM - processes sequences in both directions
    Better for understanding context from past and future
    """
    model = models.Sequential([
        layers.Bidirectional(layers.LSTM(32, return_sequences=True, 
                                        dropout=0.3, recurrent_dropout=0.3), 
                           input_shape=input_shape),
        layers.Dropout(0.4),
        layers.Bidirectional(layers.LSTM(16, return_sequences=False,
                                       dropout=0.3, recurrent_dropout=0.3)),
        layers.Dropout(0.4),
        layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(output_size, activation='linear' if task == 'regression' else 'softmax')
    ])
    return model

def create_attention_lstm(input_shape, output_size, task='regression'):
    """
    LSTM with attention mechanism
    Learns to focus on important time steps
    State-of-the-art for sequence modeling
    """
    inputs = layers.Input(shape=input_shape)
    
    # LSTM layers
    lstm_out = layers.LSTM(64, return_sequences=True, 
                          dropout=0.3, recurrent_dropout=0.3)(inputs)
    lstm_out = layers.Dropout(0.4)(lstm_out)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(lstm_out)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(64)(attention)
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention weights
    sent_representation = layers.Multiply()([lstm_out, attention])
    sent_representation = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(sent_representation)
    
    # Output layers
    dense = layers.Dense(32, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.001))(sent_representation)
    dense = layers.Dropout(0.4)(dense)
    outputs = layers.Dense(output_size, activation='linear' if task == 'regression' else 'softmax')(dense)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def create_cnn_lstm_hybrid(input_shape, output_size, task='regression'):
    """
    CNN-LSTM Hybrid: CNN extracts local features, LSTM captures temporal patterns
    Excellent for traffic data with both spatial and temporal patterns
    """
    model = models.Sequential([
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', 
                     input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        layers.Dropout(0.4),
        layers.LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
        layers.Dropout(0.4),
        layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(output_size, activation='linear' if task == 'regression' else 'softmax')
    ])
    return model

def create_transformer_model(input_shape, output_size, task='regression'):
    """
    Transformer-based model using multi-head attention
    Current state-of-the-art for sequence modeling tasks
    """
    inputs = layers.Input(shape=input_shape)
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=2, key_dim=16
    )(inputs, inputs)
    attention_output = layers.Dropout(0.3)(attention_output)
    attention_output = layers.LayerNormalization()(attention_output + inputs)
    
    # Feed forward network
    ffn = layers.Dense(64, activation='relu', 
                      kernel_regularizer=keras.regularizers.l2(0.001))(attention_output)
    ffn = layers.Dropout(0.3)(ffn)
    ffn = layers.Dense(input_shape[1], 
                      kernel_regularizer=keras.regularizers.l2(0.001))(ffn)
    ffn_output = layers.LayerNormalization()(ffn + attention_output)
    
    # Global average pooling and output
    gap = layers.GlobalAveragePooling1D()(ffn_output)
    dense = layers.Dense(32, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.001))(gap)
    dense = layers.Dropout(0.4)(dense)
    outputs = layers.Dense(output_size, activation='linear' if task == 'regression' else 'softmax')(dense)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def get_model_by_type(model_type, input_shape, output_size, task='regression', **model_kwargs):
    """Factory helper to create models based on configuration"""
    if model_type == 'rnn':
        return create_simple_rnn_model(input_shape, output_size, task=task, **model_kwargs)
    if model_type == 'lstm':
        return create_lstm_model(input_shape, output_size, task=task)
    if model_type == 'gru':
        return create_gru_model(input_shape, output_size, task=task)
    if model_type == 'bidirectional_lstm':
        return create_bidirectional_lstm(input_shape, output_size, task=task)
    if model_type == 'attention_lstm':
        return create_attention_lstm(input_shape, output_size, task=task)
    if model_type == 'cnn_lstm':
        return create_cnn_lstm_hybrid(input_shape, output_size, task=task)
    if model_type == 'transformer':
        return create_transformer_model(input_shape, output_size, task=task)
    raise ValueError(f"Unknown model type: {model_type}")

# ============================================================
# DATA PREPARATION FUNCTIONS
# ============================================================

def load_dataset(file_path):
    """Load dataset from CSV file"""
    print(f"Loading dataset from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def prepare_volume_prediction_data(csv_path, time_window=1, sequence_length=20):
    """
    Prepare network traffic volume data from Wireshark CSV
    
    Args:
        csv_path: Path to Wireshark CSV file
        time_window: Time window in seconds to aggregate traffic
        sequence_length: Number of past windows to use
    """
    # Load raw data
    df = pd.read_csv(csv_path)
    
    # Aggregate into time windows
    df['time_bin'] = (df['Time'] // time_window).astype(int)
    
    traffic = df.groupby('time_bin').agg({
        'Length': 'sum',           # Total bytes
        'No.': 'count',            # Packet count
    }).reset_index()
    
    traffic.columns = ['time_bin', 'total_bytes', 'packet_count']
    
    # Additional features
    traffic['bytes_per_packet'] = traffic['total_bytes'] / traffic['packet_count']
    
    # Select features for prediction
    features = traffic[['total_bytes', 'packet_count', 'bytes_per_packet']].values
    
    # Normalize
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_normalized) - sequence_length):
        X.append(features_normalized[i:i + sequence_length])
        y.append(features_normalized[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split train/test
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n✓ Prepared volume prediction data:")
    print(f"  Time windows: {len(traffic)}")
    print(f"  Training sequences: {len(X_train)}")
    print(f"  Testing sequences: {len(X_test)}")
    print(f"  Features per timestep: {X_train.shape[2]}")
    
    return X_train, X_test, y_train, y_test, scaler

def prepare_protocol_classification_data(csv_path, sequence_length=20):
    """
    Prepare protocol classification data from Wireshark CSV
    """
    df = pd.read_csv(csv_path)
    
    # Encode protocols
    label_encoder = LabelEncoder()
    df['protocol_encoded'] = label_encoder.fit_transform(df['Protocol'])
    
    print(f"\n✓ Protocol classes: {label_encoder.classes_}")
    print(f"  Distribution:")
    print(df['Protocol'].value_counts())
    
    # Create features
    features = df[['Length']].values
    protocols = df['protocol_encoded'].values
    
    # Normalize lengths
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_normalized) - sequence_length):
        # Combine length and protocol history
        seq_features = features_normalized[i:i + sequence_length]
        seq_protocols = protocols[i:i + sequence_length]
        
        # One-hot encode protocols
        seq_protocols_onehot = tf.keras.utils.to_categorical(
            seq_protocols, 
            num_classes=len(label_encoder.classes_)
        )
        
        # Concatenate features
        seq_combined = np.concatenate([seq_features, seq_protocols_onehot], axis=1)
        
        X.append(seq_combined)
        y.append(protocols[i + sequence_length])
    
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(label_encoder.classes_))
    
    # Split
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n✓ Prepared protocol classification data:")
    print(f"  Training sequences: {len(X_train)}")
    print(f"  Testing sequences: {len(X_test)}")
    print(f"  Features per timestep: {X_train.shape[2]}")
    
    return X_train, X_test, y_train, y_test, label_encoder, scaler

# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val, task='regression', 
                epochs=100, batch_size=32, learning_rate=0.0005,
                early_stopping_patience=10, early_stopping_min_delta=0.0001,
                reduce_lr_factor=0.5, reduce_lr_patience=5, reduce_lr_min_lr=1e-7):
    """Train model with callbacks for optimal performance"""
    
    # Compile model
    if task == 'regression':
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mae',
            metrics=['mse']
        )
    else:  # classification
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        min_delta=early_stopping_min_delta
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=reduce_lr_min_lr
    )
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1  # Show epoch-by-epoch output with loss and val_loss
    )
    
    return history

def evaluate_regression_model(model, X_test, y_test, scaler=None):
    """Evaluate volume prediction model"""
    print("\n" + "="*50)
    print("VOLUME PREDICTION EVALUATION")
    print("="*50)
    
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\nTest Set Performance:")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    
    return predictions, mse, mae, rmse

def evaluate_classification_model(model, X_test, y_test, label_encoder):
    """Evaluate protocol classification model"""
    print("\n" + "="*50)
    print("PROTOCOL CLASSIFICATION EVALUATION")
    print("="*50)
    
    predictions = model.predict(X_test)
    y_pred_classes = np.argmax(predictions, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        y_true_classes, 
        y_pred_classes, 
        target_names=label_encoder.classes_
    ))
    
    return predictions, accuracy

def plot_results(history, predictions, y_test, task='regression', model_name=None, output_dir='.', target_name=None):
    """Plot training history and predictions"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot training history
    if task == 'regression':
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_ylabel('Loss (MAE)')
    else:
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_ylabel('Accuracy')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot predictions
    if task == 'regression':
        num_samples = min(100, len(y_test))
        feature_idx = 0  # Plot first (or only) feature
        
        # Validate data variation
        y_actual = y_test[:num_samples, feature_idx]
        y_pred = predictions[:num_samples, feature_idx]
        
        actual_std = np.std(y_actual)
        pred_std = np.std(y_pred)
        
        print(f"\n📊 Plotting Statistics:")
        print(f"  Actual values - min: {y_actual.min():.6f}, max: {y_actual.max():.6f}, std: {actual_std:.6f}")
        print(f"  Predicted values - min: {y_pred.min():.6f}, max: {y_pred.max():.6f}, std: {pred_std:.6f}")
        
        if actual_std < 1e-6:
            print(f"  ⚠️  WARNING: Actual values have no variation (std={actual_std:.6f})!")
            print(f"  This indicates a data preprocessing problem.")
        if pred_std < 1e-6:
            print(f"  ⚠️  WARNING: Predictions have no variation (std={pred_std:.6f})!")
            print(f"  The model is predicting constant values.")
        
        feature_name = target_name if target_name else f"Feature {feature_idx}"
        axes[1].plot(y_actual, label='Actual', linewidth=2, marker='o', markersize=4)
        axes[1].plot(y_pred, label='Predicted', linewidth=2, marker='x', alpha=0.7, markersize=4)
        axes[1].set_title(f'Predictions vs Actual ({feature_name})')
        axes[1].set_ylabel('Normalized Value')
    else:
        num_samples = min(100, len(y_test))
        y_true_classes = np.argmax(y_test[:num_samples], axis=1)
        y_pred_classes = np.argmax(predictions[:num_samples], axis=1)
        axes[1].plot(y_true_classes, label='Actual Protocol', linewidth=2, marker='o')
        axes[1].plot(y_pred_classes, label='Predicted Protocol', linewidth=2, marker='x', alpha=0.7)
        axes[1].set_title('Protocol Predictions vs Actual')
        axes[1].set_ylabel('Protocol Class')
    
    axes[1].set_xlabel('Sample')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{task}_results.png" if model_name is None else f"{task}_results_{model_name}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved as '{plot_path}'")
    plt.show()

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*70)
    print("ADVANCED NETWORK TRAFFIC PREDICTION WITH MODERN RNN ARCHITECTURES")
    print("="*70)
    
    # ==================== CONFIGURATION ====================
    # Dataset paths
    data_dir = os.path.join('.', 'Dataset')
    volume_dataset_path = os.path.join(data_dir, 'Midterm_53_group.csv')
    protocol_dataset_path = os.path.join(data_dir, 'Midterm_53_group.csv')
    protocol_column = 'Protocol'  # Column name containing protocol labels
    
    # Model selection - RUN ONE MODEL AT A TIME
    model_type = 'lstm'  # Options: 'rnn', 'lstm', 'gru', 'bidirectional_lstm', 
                       #          'attention_lstm', 'cnn_lstm', 'transformer'
    
    # ========== DATA PREPARATION HYPERPARAMETERS ==========
    sequence_length = 20      # Length of input sequences
    split_ratio = 0.7         # Train/test split ratio
    target_column = 'total_bytes'  # Target column for volume prediction (None = predict all numeric columns)
                              # Options: 'Length', 'Time', 'No.', or None
    
    # ========== RNN MODEL ARCHITECTURE HYPERPARAMETERS ==========
    rnn1_units = 128             # Units in first RNN layer
    rnn2_units = 64              # Units in second RNN layer
    dense_units = 32             # Units in dense layer
    rnn_dropout = 0.2            # Dropout rate for RNN layers
    rnn_recurrent_dropout = 0.2  # Recurrent dropout rate for RNN layers
    dropout1 = 0.2               # Dropout after first RNN layer
    dropout2 = 0.2               # Dropout after second RNN layer
    dropout3 = 0.2               # Dropout after dense layer
    l2_reg = 0.01               # L2 regularization strength
    
    # ========== TRAINING HYPERPARAMETERS ==========
    epochs = 5                  # Maximum number of epochs
    batch_size = 32               # Batch size for training
    learning_rate = 0.001        # Initial learning rate
    
    # ========== CALLBACK HYPERPARAMETERS ==========
    early_stopping_patience = 10      # Early stopping patience
    early_stopping_min_delta = 0.0001 # Minimum change to qualify as improvement
    reduce_lr_factor = 0.5            # Factor to reduce learning rate
    reduce_lr_patience = 5            # Patience for learning rate reduction
    reduce_lr_min_lr = 1e-7           # Minimum learning rate
    
    results_dir = os.path.join('.', 'Results')
    os.makedirs(results_dir, exist_ok=True)
    models_dir = os.path.join('.', 'Models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Task selection
    run_volume_prediction = True
    run_protocol_classification = False  # Set to True to run classification task
    
    # ========== PRINT ALL HYPERPARAMETERS ==========
    print("\n" + "="*70)
    print("HYPERPARAMETER CONFIGURATION")
    print("="*70)
    print(f"\nModel Type: {model_type}")
    print(f"\nData Preparation:")
    print(f"  - Sequence Length: {sequence_length}")
    print(f"  - Train/Test Split Ratio: {split_ratio}")
    print(f"  - Target Column: {target_column if target_column else 'All numeric columns'}")
    print(f"\nRNN Model Architecture:")
    print(f"  - RNN Layer 1 Units: {rnn1_units}")
    print(f"  - RNN Layer 2 Units: {rnn2_units}")
    print(f"  - Dense Layer Units: {dense_units}")
    print(f"  - RNN Dropout: {rnn_dropout}")
    print(f"  - RNN Recurrent Dropout: {rnn_recurrent_dropout}")
    print(f"  - Dropout 1 (after RNN1): {dropout1}")
    print(f"  - Dropout 2 (after RNN2): {dropout2}")
    print(f"  - Dropout 3 (after Dense): {dropout3}")
    print(f"  - L2 Regularization: {l2_reg}")
    print(f"\nTraining:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"\nCallbacks:")
    print(f"  - Early Stopping Patience: {early_stopping_patience}")
    print(f"  - Early Stopping Min Delta: {early_stopping_min_delta}")
    print(f"  - Reduce LR Factor: {reduce_lr_factor}")
    print(f"  - Reduce LR Patience: {reduce_lr_patience}")
    print(f"  - Reduce LR Min LR: {reduce_lr_min_lr}")
    print("="*70)
    # =======================================================
    
    # TASK 1: Volume Prediction
    if run_volume_prediction and os.path.exists(volume_dataset_path):
        print("\n" + "="*70)
        print("TASK 1: NETWORK TRAFFIC VOLUME PREDICTION")
        print("="*70)
        
        df = load_dataset(volume_dataset_path)
        if df is not None:
            result = prepare_volume_prediction_data(
                df, sequence_length, split_ratio, target_column=target_column
            )
            
            if result is None or result[0] is None:
                print("Failed to prepare data. Check the warnings above.")
                return
            
            X_train, X_test, y_train, y_test, scaler = result
            
            if X_train is not None:
                input_shape = (X_train.shape[1], X_train.shape[2])
                output_size = y_train.shape[1]

                print(f"\nCreating {model_type} model...")
                try:
                    # Pass RNN-specific hyperparameters
                    model = get_model_by_type(
                        model_type, input_shape, output_size, task='regression',
                        rnn1_units=rnn1_units, rnn2_units=rnn2_units, dense_units=dense_units,
                        rnn_dropout=rnn_dropout, rnn_recurrent_dropout=rnn_recurrent_dropout,
                        dropout1=dropout1, dropout2=dropout2, dropout3=dropout3,
                        l2_reg=l2_reg
                    )
                except ValueError as err:
                    print(f"Error creating model: {err}")
                    return
                
                print("\nModel Summary:")
                model.summary()
                
                # Train model
                history = train_model(
                    model, X_train, y_train, X_test, y_test,
                    task='regression', epochs=epochs, batch_size=batch_size,
                    learning_rate=learning_rate,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    reduce_lr_factor=reduce_lr_factor,
                    reduce_lr_patience=reduce_lr_patience,
                    reduce_lr_min_lr=reduce_lr_min_lr
                )
                
                # Evaluate
                predictions, mse, mae, rmse = evaluate_regression_model(
                    model, X_test, y_test, scaler
                )
                
                # Plot results
                plot_results(
                    history, predictions, y_test,
                    task='regression', model_name=model_type, output_dir=results_dir,
                    target_name=target_column
                )
                
                # Save model
                model_filename = f'volume_prediction_{model_type}.h5'
                model_path = os.path.join(models_dir, model_filename)
                model.save(model_path)
                print(f"\nModel saved as '{model_path}'")
    
    # TASK 2: Protocol Classification
    if run_protocol_classification and os.path.exists(protocol_dataset_path):
        print("\n" + "="*70)
        print("TASK 2: PROTOCOL CLASSIFICATION")
        print("="*70)
        
        df = load_dataset(protocol_dataset_path)
        if df is not None:
            classification_result = prepare_protocol_classification_data(
                df, protocol_column, sequence_length, split_ratio
            )
            
            if not classification_result or classification_result[0] is None:
                print("Skipping protocol classification - data preparation failed (check protocol column).")
            else:
                X_train, X_test, y_train, y_test, label_encoder, _ = classification_result
                
                input_shape = (X_train.shape[1], X_train.shape[2])
                output_size = y_train.shape[1]
                
                print(f"\nCreating {model_type} model for classification...")
                try:
                    # Pass RNN-specific hyperparameters
                    model = get_model_by_type(
                        model_type, input_shape, output_size, task='classification',
                        rnn1_units=rnn1_units, rnn2_units=rnn2_units, dense_units=dense_units,
                        rnn_dropout=rnn_dropout, rnn_recurrent_dropout=rnn_recurrent_dropout,
                        dropout1=dropout1, dropout2=dropout2, dropout3=dropout3,
                        l2_reg=l2_reg
                    )
                except ValueError as err:
                    print(f"Error creating model: {err}")
                    return
                
                print("\nModel Summary:")
                model.summary()
                
                # Train model
                history = train_model(
                    model, X_train, y_train, X_test, y_test,
                    task='classification', epochs=epochs, batch_size=batch_size,
                    learning_rate=learning_rate,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    reduce_lr_factor=reduce_lr_factor,
                    reduce_lr_patience=reduce_lr_patience,
                    reduce_lr_min_lr=reduce_lr_min_lr
                )
                
                # Evaluate
                predictions, accuracy = evaluate_classification_model(
                    model, X_test, y_test, label_encoder
                )
                
                # Plot results
                plot_results(
                    history, predictions, y_test,
                    task='classification', model_name=model_type, output_dir=results_dir
                )
                
                # Save model
                model_filename = f'protocol_classification_{model_type}.h5'
                model_path = os.path.join(models_dir, model_filename)
                model.save(model_path)
                print(f"\nModel saved as '{model_path}'")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()