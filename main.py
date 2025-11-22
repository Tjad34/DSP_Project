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
import pickle
from datasets import load_dataset

def load_huggingface_dataset(use_cache=True, cache_file='./data/cached_dataset.pkl'):
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        return df
    try:
        dataset = load_dataset("abmallick/network-traffic-anomaly", split='train', streaming=True)
        rows = []
        for i, row in enumerate(dataset):
            rows.append(row)
            if i >= 149999:
                break
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        return df
    except Exception as e:
        return None

def rnn_model(input_shape, output_size, task):
    model = models.Sequential([
        layers.SimpleRNN(64, return_sequences=True, input_shape=input_shape,dropout=0.3),
        layers.Dropout(0.4),
        layers.SimpleRNN(32, return_sequences=False,dropout=0.3),
        layers.Dropout(0.4),
        layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(output_size, activation='linear' if task == 'regression' else 'softmax')
    ])
    return model

def lstm_model(input_shape, output_size, task):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape,dropout=0.3),
        layers.Dropout(0.4),
        layers.LSTM(32, return_sequences=False,dropout=0.3),
        layers.Dropout(0.4),
        layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(output_size, activation='linear' if task == 'regression' else 'softmax')
    ])
    return model

def prepare_volume_prediction_data(df, sequence_length=20, train_ratio=0.6, val_ratio=0.2):
    if isinstance(df, str):
        df = pd.read_csv(df)
    features = df[[
        'TotLen Fwd Pkts',
        'TotLen Bwd Pkts',
        'Flow Byts/s',
        'Flow Pkts/s'
    ]].copy()
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    features = np.log1p(features)
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(features)
    X, y = [], []
    for i in range(len(data_normalized) - sequence_length):
        X.append(data_normalized[i:i + sequence_length])
        y.append(data_normalized[i + sequence_length])
    X = np.array(X)
    y = np.array(y)
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def prepare_protocol_classification_data(df, sequence_length=20, train_ratio=0.6, val_ratio=0.2, max_samples=50000, step=5):
    if isinstance(df, str):
        df = pd.read_csv(df)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    label_encoder = LabelEncoder()
    df['protocol_encoded'] = label_encoder.fit_transform(df['Protocol'].astype(str))
    features_df = df[[
        'TotLen Fwd Pkts',
        'TotLen Bwd Pkts',
        'Flow Byts/s',
        'Flow Pkts/s',
        'Flow Duration'
    ]].copy()
    features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features_df)
    df_clean = df.loc[features_df.index].copy()
    X, y = [], []
    for i in range(0, len(features_normalized) - sequence_length, step):
        seq = np.column_stack([
            features_normalized[i:i + sequence_length],
            df_clean.iloc[i:i + sequence_length]['protocol_encoded'].values.reshape(-1, 1)
        ])
        X.append(seq)
        y.append(df_clean.iloc[i + sequence_length]['protocol_encoded'])
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(label_encoder.classes_))
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler

def train_model(model, X_train, y_train, X_val, y_val, task, epochs, batch_size, learning_rate, early_stopping_patience, early_stopping_min_delta, reduce_lr_factor, reduce_lr_patience, reduce_lr_min_lr, class_weight=None):
    if task == 'regression':
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mae', metrics=['mse'])
    else:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, min_delta=early_stopping_min_delta)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=reduce_lr_min_lr)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight,
        verbose=1
    )
    return history

def evaluate_regression_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return predictions, mse, mae, rmse

def evaluate_classification_model(model, X_test, y_test, label_encoder=None):
    predictions = model.predict(X_test)
    y_pred_classes = np.argmax(predictions, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    unique_classes = np.unique(y_true_classes)
    if label_encoder is not None:
        target_names = [label_encoder.classes_[i] for i in unique_classes]
    else:
        target_names = ['Normal', 'Anomaly'] if len(unique_classes) == 2 else [f'Class {i}' for i in unique_classes]
    print(classification_report(
        y_true_classes, 
        y_pred_classes, 
        labels=unique_classes,
        target_names=target_names
    ))
    return predictions, accuracy

def plot_results(history, predictions, y_test, task, output_dir='.', model_name=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    if task == 'regression':
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_ylabel('Loss (MSE)')
    else:
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True)
    num_samples = min(100, len(y_test))
    if task == 'regression':
        axes[1].plot(y_test[:num_samples, 0], label='Actual', linewidth=2, marker='o')
        axes[1].plot(predictions[:num_samples, 0], label='Predicted', linewidth=2, marker='x', alpha=0.7)
        axes[1].set_title('Predictions vs Actual (First Feature)')
        axes[1].set_ylabel('Normalized Value')
    else:
        y_true = np.argmax(y_test[:num_samples], axis=1)
        y_pred = np.argmax(predictions[:num_samples], axis=1)
        axes[1].plot(y_true, label='Actual Protocol', linewidth=2, marker='o')
        axes[1].plot(y_pred, label='Predicted Protocol', linewidth=2, marker='x', alpha=0.7)
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
    plt.show()

def prepare_session_anomaly_detection(csv_path, sequence_length=20, train_ratio=0.6, val_ratio=0.2, sample_size=100000):
    df = pd.read_csv(csv_path)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    labels = df['label'].values
    feature_cols = [col for col in df.columns if col not in ['label', 'unique_link_mark']]
    features_df = df[feature_cols].copy()
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    nan_threshold = 0.5
    features_df = features_df.loc[:, features_df.isnull().mean() < nan_threshold]
    features_df = features_df.fillna(0)
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    features_selected = selector.fit_transform(features_df)
    variances = selector.variances_
    top_indices = np.argsort(variances)[-30:][::-1]
    features_final = features_selected[:, top_indices]
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features_final)
    X, y = [], []
    for i in range(len(features_normalized) - sequence_length):
        X.append(features_normalized[i:i + sequence_length])
        y.append(labels[i + sequence_length])
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=2)
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def main():
    csv_path = './Dataset/session_based_dataset.csv'
    model_type = 'lstm'
    sequence_length = 20
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    early_stopping_patience = 10
    early_stopping_min_delta = 0.0001
    reduce_lr_factor = 0.5
    reduce_lr_patience = 5
    reduce_lr_min_lr = 1e-7
    run_volume_prediction = True
    run_protocol_classification = True
    results_dir = os.path.join('.', 'Results')
    os.makedirs(results_dir, exist_ok=True)
    models_dir = os.path.join('.', 'Models')
    os.makedirs(models_dir, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_session_anomaly_detection(
        csv_path, 
        sequence_length=20,
        sample_size=100000
    )
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_size = 2
    if model_type == 'rnn':
        model = rnn_model(input_shape, output_size,task='classification')
    elif model_type == 'lstm':
        model = lstm_model(input_shape, output_size,task='classification')
    from sklearn.utils.class_weight import compute_class_weight
    y_train_classes = np.argmax(y_train, axis=1)
    unique_classes = np.unique(y_train_classes)
    if len(unique_classes) > 1:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train_classes
        )
        class_weight_dict = dict(zip(unique_classes, class_weights))
    else:
        class_weight_dict = None
    history = train_model(model, X_train, y_train, X_val, y_val, task='classification', epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, early_stopping_patience=early_stopping_patience, early_stopping_min_delta=early_stopping_min_delta, reduce_lr_factor=reduce_lr_factor, reduce_lr_patience=reduce_lr_patience, reduce_lr_min_lr=reduce_lr_min_lr,class_weight=class_weight_dict)
    predictions, accuracy = evaluate_classification_model(model, X_test, y_test, label_encoder=None)
    y_pred_classes = np.argmax(predictions, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_true_classes, y_pred_classes, 
                                target_names=['Benign', 'Anomaly']))
    print(confusion_matrix(y_true_classes, y_pred_classes))
    plot_results(history, predictions, y_test, task='classification', 
                output_dir=results_dir, model_name=model_type)
    model_path = os.path.join(models_dir, f'anomaly_{model_type}.h5')
    model.save(model_path)

if __name__ == "__main__":
    main()

