import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import itertools

def build_model_with_config(input_dim, model_type, config):
    if model_type == 'dense':
        model = models.Sequential([
            layers.Dense(config['units_1'], activation='relu', input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(config['dropout_1']),
            layers.Dense(config['units_2'], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(config['dropout_2']),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(2, activation='softmax')
        ], name=f'Dense_{config["units_1"]}_{config["units_2"]}')
    elif model_type == 'lstm':
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Reshape((1, input_dim))(inputs)
        x = layers.LSTM(config['units_1'], return_sequences=False)(x)
        x = layers.Dropout(config['dropout_1'])(x)
        x = layers.Dense(config['units_2'], activation='relu')(x)
        x = layers.Dropout(config['dropout_2'])(x)
        outputs = layers.Dense(2, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs, 
                           name=f'LSTM_{config["units_1"]}_{config["units_2"]}')
    return model

def quick_hyperparameter_search(X_train, y_train, X_val, y_val, input_dim, model_type, class_weight):
    param_grid = {
        'units_1': [128, 256],
        'units_2': [64, 128],
        'dropout_1': [0.3, 0.4],
        'dropout_2': [0.2, 0.3],
        'learning_rate': [0.0005, 0.001]
    }
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_val_acc = 0
    best_config = None
    results = []
    for i, config in enumerate(combinations, 1):
        model = build_model_with_config(input_dim, model_type, config)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=256,
            class_weight=class_weight,
            verbose=0
        )
        val_acc = max(history.history['val_accuracy'])
        results.append({
            'config': config,
            'val_acc': val_acc
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = config
    return best_config, best_val_acc, results

def simple_tuning(X_train, y_train, X_val, y_val, input_dim, model_type, class_weight):
    configs = [
        {'units_1': 128, 'units_2': 64, 'dropout_1': 0.3, 'dropout_2': 0.2, 'learning_rate': 0.001},
        {'units_1': 256, 'units_2': 128, 'dropout_1': 0.4, 'dropout_2': 0.3, 'learning_rate': 0.001},
        {'units_1': 256, 'units_2': 128, 'dropout_1': 0.4, 'dropout_2': 0.3, 'learning_rate': 0.0005},
    ]
    best_val_acc = 0
    best_config = None
    for i, config in enumerate(configs, 1):
        model = build_model_with_config(input_dim, model_type, config)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=256,
            class_weight=class_weight,
            verbose=0
        )
        val_acc = max(history.history['val_accuracy'])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = config
    return best_config, best_val_acc
