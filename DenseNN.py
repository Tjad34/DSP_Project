import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

def prepare_anomaly_data_simple(csv_path, test_size=0.2, val_size=0.1, max_samples=100000):
    """
    Simple approach WITHOUT sequences - treat each flow independently
    This is better when flow labels are independent of previous flows
    """
    print(f"\n{'='*70}")
    print("LOADING AND PREPARING DATA")
    print(f"{'='*70}")
    
    df = pd.read_csv(csv_path)
    
    print(f"\nOriginal dataset: {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Anomaly %: {(df['label'].sum()/len(df))*100:.2f}%")
    
    # Sample if needed
    if len(df) > max_samples:
        # Stratified sampling to maintain balance
        normal = df[df['label']==0].sample(n=max_samples//2, random_state=42)
        anomaly = df[df['label']==1].sample(n=max_samples//2, random_state=42)
        df = pd.concat([normal, anomaly]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"\nSampled to {len(df)} samples (maintaining 50/50 balance)")
    
    # Get features and labels
    labels = df['label'].values
    feature_cols = [col for col in df.columns if col not in ['label', 'unique_link_mark', 'Unnamed: 0']]
    features = df[feature_cols].values
    
    # Handle any inf/nan
    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"Using {features.shape[1]} features per flow")
    
    # Normalize using StandardScaler
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # NO SEQUENCES - just use raw features
    X = features_normalized
    y = labels
    
    print(f"\nData prepared: {len(X)} flows")
    print(f"Label distribution: Normal={np.sum(y==0)}, Anomaly={np.sum(y==1)}")
    print(f"Anomaly %: {(np.sum(y==1)/len(y))*100:.2f}%")
    
    # Split: train/val/test with shuffle
    from sklearn.model_selection import train_test_split
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=42, stratify=y
    )
    
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_ratio), random_state=42, stratify=y_temp
    )
    
    # Convert to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)
    
    print(f"\n{'='*70}")
    print("DATA SPLIT (STRATIFIED)")
    print(f"{'='*70}")
    print(f"Train: {len(X_train):6d} samples | Normal: {np.sum(y_train==0):6d} | Anomaly: {np.sum(y_train==1):6d} | ({np.sum(y_train==1)/len(y_train)*100:.1f}% anomaly)")
    print(f"Val:   {len(X_val):6d} samples | Normal: {np.sum(y_val==0):6d} | Anomaly: {np.sum(y_val==1):6d} | ({np.sum(y_val==1)/len(y_val)*100:.1f}% anomaly)")
    print(f"Test:  {len(X_test):6d} samples | Normal: {np.sum(y_test==0):6d} | Anomaly: {np.sum(y_test==1):6d} | ({np.sum(y_test==1)/len(y_test)*100:.1f}% anomaly)")
    
    return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, y_train, y_val, y_test, scaler

def build_dense_model(input_dim, output_size=2):
    """Build Dense Neural Network - better for independent samples"""
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
        layers.Dense(output_size, activation='softmax')
    ])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, 
                learning_rate=0.001, class_weight=None):
    """Train the model"""
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    print(f"\n{'='*70}")
    print("TRAINING MODEL")
    print(f"{'='*70}\n")
    
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

def evaluate_model(model, X_test, y_test_cat, y_test_raw):
    """Comprehensive evaluation"""
    
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    y_true_classes = np.argmax(y_test_cat, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes)
    recall = recall_score(y_true_classes, y_pred_classes)
    f1 = f1_score(y_true_classes, y_pred_classes)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (of predicted anomalies, how many were correct)")
    print(f"  Recall:    {recall:.4f} (of actual anomalies, how many were detected)")
    print(f"  F1-Score:  {f1:.4f}")
    
    # ROC-AUC
    try:
        auc = roc_auc_score(y_true_classes, y_pred_proba[:, 1])
        print(f"  ROC-AUC:   {auc:.4f}")
    except:
        auc = None
        print(f"  ROC-AUC:   N/A")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                                target_names=['Normal', 'Anomaly'], digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(f"Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"      Anomaly   {cm[1,0]:5d}   {cm[1,1]:5d}")
    
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    print(f"\nTrue Negatives:  {tn:5d} (correctly identified normal)")
    print(f"False Positives: {fp:5d} (normal misclassified as anomaly)")
    print(f"False Negatives: {fn:5d} (anomaly misclassified as normal) ⚠️")
    print(f"True Positives:  {tp:5d} (correctly identified anomaly)")
    
    if tp + fn > 0:
        detection_rate = tp / (tp + fn)
        print(f"\nAnomaly Detection Rate: {detection_rate:.4f} ({tp}/{tp+fn} anomalies detected)")
    
    return y_pred_classes, y_pred_proba, accuracy, cm, auc

def plot_results(history, y_true, y_pred, y_pred_proba, cm, auc, output_dir='./Results'):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Training Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training & Validation Accuracy', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training & Validation Loss', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision & Recall
    ax3 = fig.add_subplot(gs[0, 2])
    if 'precision' in history.history:
        ax3.plot(history.history['precision'], label='Train Precision', linewidth=2)
        ax3.plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    if 'recall' in history.history:
        ax3.plot(history.history['recall'], label='Train Recall', linewidth=2, linestyle='--')
        ax3.plot(history.history['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision & Recall', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
    ax4.figure.colorbar(im, ax=ax4)
    ax4.set_title('Confusion Matrix', fontweight='bold')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Normal', 'Anomaly'])
    ax4.set_yticklabels(['Normal', 'Anomaly'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')
    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')
    
    # Plot 5: Prediction Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    normal_scores = y_pred_proba[y_true==0, 1]
    anomaly_scores = y_pred_proba[y_true==1, 1]
    ax5.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue')
    ax5.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red')
    ax5.set_xlabel('Anomaly Score (Probability)')
    ax5.set_ylabel('Count')
    ax5.set_title('Prediction Score Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: ROC Curve
    ax6 = fig.add_subplot(gs[1, 2])
    if auc is not None:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        ax6.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={auc:.3f})')
        ax6.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax6.set_xlabel('False Positive Rate')
        ax6.set_ylabel('True Positive Rate')
        ax6.set_title('ROC Curve', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'ROC curve unavailable', ha='center', va='center')
        ax6.set_title('ROC Curve', fontweight='bold')
    
    # Plot 7: Sample Predictions
    ax7 = fig.add_subplot(gs[2, :])
    num_samples = min(300, len(y_true))
    sample_true = y_true[:num_samples]
    sample_pred = y_pred[:num_samples]
    x_axis = np.arange(num_samples)
    
    # Plot actual
    normal_mask = sample_true == 0
    anomaly_mask = sample_true == 1
    ax7.scatter(x_axis[normal_mask], sample_true[normal_mask], 
               c='lightblue', label='Actual Normal', alpha=0.7, s=30, marker='o')
    ax7.scatter(x_axis[anomaly_mask], sample_true[anomaly_mask], 
               c='lightcoral', label='Actual Anomaly', alpha=0.7, s=30, marker='o')
    
    # Plot predictions
    pred_normal = sample_pred == 0
    pred_anomaly = sample_pred == 1
    ax7.scatter(x_axis[pred_normal], sample_pred[pred_normal] - 0.05, 
               c='blue', label='Pred Normal', alpha=0.5, s=20, marker='_')
    ax7.scatter(x_axis[pred_anomaly], sample_pred[pred_anomaly] + 0.05, 
               c='red', label='Pred Anomaly', alpha=0.5, s=20, marker='_')
    
    ax7.set_xlabel('Sample Index')
    ax7.set_ylabel('Class')
    ax7.set_title(f'Predictions vs Actual (First {num_samples} samples)', fontweight='bold')
    ax7.set_yticks([0, 1])
    ax7.set_yticklabels(['Normal', 'Anomaly'])
    ax7.legend(loc='upper right')
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'anomaly_detection_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to '{plot_path}'")
    plt.show()

def main():
    print(f"\n{'='*70}")
    print("NETWORK TRAFFIC ANOMALY DETECTION")
    print(f"{'='*70}")
    
    # Configuration
    csv_path = './Dataset/session_based_dataset.csv'
    batch_size = 256  # Larger batches for dense network
    epochs = 50
    learning_rate = 0.001
    max_samples = 100000
    
    # Prepare data
    result = prepare_anomaly_data_simple(
        csv_path,
        max_samples=max_samples
    )
    
    X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, y_train_raw, y_val_raw, y_test_raw, scaler = result
    
    # Build model
    input_dim = X_train.shape[1]  # Number of features
    print(f"\n{'='*70}")
    print(f"Building Dense Neural Network")
    print(f"Input features: {input_dim}")
    print(f"{'='*70}")
    
    model = build_dense_model(input_dim)
    model.summary()
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_raw),
        y=y_train_raw
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")
    
    # Train
    history = train_model(
        model, X_train, y_train_cat, X_val, y_val_cat,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        class_weight=class_weight_dict
    )
    
    # Evaluate
    y_pred, y_pred_proba, accuracy, cm, auc = evaluate_model(
        model, X_test, y_test_cat, y_test_raw
    )
    
    # Plot
    plot_results(history, y_test_raw, y_pred, y_pred_proba, cm, auc)
    
    # Save model
    os.makedirs('./Models', exist_ok=True)
    model_path = './Models/anomaly_dense.keras'
    model.save(model_path)
    print(f"\n✓ Model saved to '{model_path}'")
    
    print(f"\n{'='*70}")
    print("✓ TRAINING COMPLETE!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()