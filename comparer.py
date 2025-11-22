import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import os

# ============================================================
# DATA PREPARATION (Same for all models)
# ============================================================

def prepare_data(csv_path, max_samples=100000):
    """Prepare data - same for all architectures"""
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    
    df = pd.read_csv(csv_path)
    print(f"Original dataset: {len(df)} samples")
    
    # Stratified sampling to maintain 50/50 balance
    if len(df) > max_samples:
        normal = df[df['label']==0].sample(n=max_samples//2, random_state=42)
        anomaly = df[df['label']==1].sample(n=max_samples//2, random_state=42)
        df = pd.concat([normal, anomaly]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Using {len(df)} samples (50/50 balance)")
    
    # Extract features and labels
    labels = df['label'].values
    feature_cols = [col for col in df.columns if col not in ['label', 'unique_link_mark', 'Unnamed: 0']]
    features = df[feature_cols].values
    
    # Handle inf/nan
    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Normalize
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    print(f"Features: {features_normalized.shape[1]}")
    
    # Split data (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_normalized, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.67, random_state=42, stratify=y_temp
    )
    
    # Convert to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)
    
    print(f"\nData split:")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, y_train, y_val, y_test

# ============================================================
# MODEL ARCHITECTURES
# ============================================================

def build_dense_nn(input_dim):
    """Dense Neural Network (Baseline - Current Best)"""
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')
    ], name='Dense_NN')
    return model

def build_rnn(input_dim):
    """Simple RNN - treat as single timestep"""
    inputs = layers.Input(shape=(input_dim,))
    # Reshape to (batch, 1, features) for RNN
    x = layers.Reshape((1, input_dim))(inputs)
    x = layers.SimpleRNN(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name='RNN')
    return model

def build_lstm(input_dim):
    """LSTM - treat as single timestep"""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Reshape((1, input_dim))(inputs)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name='LSTM')
    return model

def build_gru(input_dim):
    """GRU - treat as single timestep"""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Reshape((1, input_dim))(inputs)
    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name='GRU')
    return model

def build_bidirectional_lstm(input_dim):
    """Bidirectional LSTM"""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Reshape((1, input_dim))(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name='Bidirectional_LSTM')
    return model

# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                       class_weight, epochs=50, batch_size=256):
    """Train and evaluate a model"""
    
    model_name = model.name
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*70}")
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0)
    
    # Train
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight,
        verbose=0  # Silent training
    )
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    except:
        auc = 0.5
    
    # Count parameters
    total_params = model.count_params()
    
    results = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'training_time': training_time,
        'params': total_params,
        'epochs_trained': len(history.history['loss']),
        'history': history
    }
    
    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Training:  {training_time:.1f}s")
    print(f"  Epochs:    {len(history.history['loss'])}")
    print(f"  Params:    {total_params:,}")
    
    return results

# ============================================================
# VISUALIZATION
# ============================================================

def plot_comparison(all_results, output_dir='./Results'):
    """Create comprehensive comparison plots"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    models = [r['model'] for r in all_results]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Architecture Comparison: Network Anomaly Detection', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Comparison
    accuracies = [r['accuracy'] for r in all_results]
    colors = ['green' if acc > 0.99 else 'orange' if acc > 0.95 else 'red' for acc in accuracies]
    axes[0, 0].barh(models, accuracies, color=colors)
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_xlim([0.9, 1.0])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(v - 0.01, i, f'{v:.4f}', va='center', ha='right', fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. Precision vs Recall
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    x = np.arange(len(models))
    width = 0.35
    axes[0, 1].bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
    axes[0, 1].bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision & Recall')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0.9, 1.0])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. ROC-AUC
    aucs = [r['auc'] for r in all_results]
    axes[0, 2].barh(models, aucs, color='purple', alpha=0.7)
    axes[0, 2].set_xlabel('ROC-AUC')
    axes[0, 2].set_title('ROC-AUC Score')
    axes[0, 2].set_xlim([0.9, 1.0])
    for i, v in enumerate(aucs):
        axes[0, 2].text(v - 0.01, i, f'{v:.4f}', va='center', ha='right', fontweight='bold')
    axes[0, 2].grid(axis='x', alpha=0.3)
    
    # 4. Training Time
    times = [r['training_time'] for r in all_results]
    axes[1, 0].barh(models, times, color='skyblue')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_title('Training Time')
    for i, v in enumerate(times):
        axes[1, 0].text(v + 1, i, f'{v:.1f}s', va='center', fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 5. Model Parameters
    params = [r['params']/1000 for r in all_results]  # Convert to thousands
    axes[1, 1].barh(models, params, color='coral')
    axes[1, 1].set_xlabel('Parameters (thousands)')
    axes[1, 1].set_title('Model Complexity')
    for i, v in enumerate(params):
        axes[1, 1].text(v + 5, i, f'{v:.0f}K', va='center', fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    # 6. F1-Score Comparison
    f1_scores = [r['f1'] for r in all_results]
    axes[1, 2].barh(models, f1_scores, color='mediumseagreen')
    axes[1, 2].set_xlabel('F1-Score')
    axes[1, 2].set_title('F1-Score (Harmonic Mean)')
    axes[1, 2].set_xlim([0.9, 1.0])
    for i, v in enumerate(f1_scores):
        axes[1, 2].text(v - 0.01, i, f'{v:.4f}', va='center', ha='right', fontweight='bold')
    axes[1, 2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'architecture_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to '{output_dir}/architecture_comparison.png'")
    plt.show()

def create_summary_table(all_results):
    """Create a summary table of results"""
    print(f"\n{'='*120}")
    print("SUMMARY TABLE: ALL ARCHITECTURES")
    print(f"{'='*120}")
    
    header = f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10} {'Time(s)':>10} {'Params':>12}"
    print(header)
    print("-" * 120)
    
    for r in all_results:
        row = f"{r['model']:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['auc']:>10.4f} {r['training_time']:>10.1f} {r['params']:>12,}"
        print(row)
    
    print("=" * 120)
    
    # Find best model
    best_acc = max(all_results, key=lambda x: x['accuracy'])
    best_f1 = max(all_results, key=lambda x: x['f1'])
    fastest = min(all_results, key=lambda x: x['training_time'])
    
    print(f"\n🏆 Best Accuracy:  {best_acc['model']} ({best_acc['accuracy']:.4f})")
    print(f"🏆 Best F1-Score:  {best_f1['model']} ({best_f1['f1']:.4f})")
    print(f"⚡ Fastest:        {fastest['model']} ({fastest['training_time']:.1f}s)")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*70)
    print("ARCHITECTURE COMPARISON FOR NETWORK ANOMALY DETECTION")
    print("="*70)
    
    # Configuration
    csv_path = './Dataset/session_based_dataset.csv'
    max_samples = 100000
    epochs = 50
    batch_size = 256
    
    # Prepare data (same for all models)
    X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, y_train, y_val, y_test = prepare_data(
        csv_path, max_samples
    )
    
    input_dim = X_train.shape[1]
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Define architectures to test
    architectures = {
        'Dense_NN': build_dense_nn,
        'RNN': build_rnn,
        'LSTM': build_lstm,
        'GRU': build_gru,
        'Bidirectional_LSTM': build_bidirectional_lstm
    }
    
    # Train and evaluate all models
    all_results = []
    
    for arch_name, build_func in architectures.items():
        print(f"\n{'#'*70}")
        print(f"Testing: {arch_name}")
        print(f"{'#'*70}")
        
        model = build_func(input_dim)
        
        results = train_and_evaluate(
            model, X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat,
            class_weight_dict, epochs, batch_size
        )
        
        all_results.append(results)
        
        # Save model
        os.makedirs('./Models', exist_ok=True)
        model.save(f'./Models/anomaly_{arch_name.lower()}.keras')
        print(f"✓ Model saved to './Models/anomaly_{arch_name.lower()}.keras'")
    
    # Create visualizations and summary
    create_summary_table(all_results)
    plot_comparison(all_results)
    
    print(f"\n{'='*70}")
    print("✓ ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()