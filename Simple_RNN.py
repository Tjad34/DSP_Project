import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Simple RNN implementation
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.last_inputs = inputs
        self.last_hs = {0: h}
        
        # Forward pass through time
        for t, x in enumerate(inputs):
            x = x.reshape(-1, 1)
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            self.last_hs[t + 1] = h
        
        # Output
        y = np.dot(self.Why, h) + self.by
        return y, h
    
    def backward(self, target, y, h):
        # Calculate output gradient
        dy = 2 * (y - target)
        
        # Update output weights
        dWhy = np.dot(dy, h.T)
        dby = dy
        
        # Clip gradients to prevent exploding gradients
        dWhy = np.clip(dWhy, -5, 5)
        dby = np.clip(dby, -5, 5)
        
        self.Why -= self.learning_rate * dWhy
        self.by -= self.learning_rate * dby
        
    def train_step(self, X, y):
        output, h = self.forward(X)
        self.backward(y, output, h)
        loss = np.mean((output - y) ** 2)
        return loss

def load_dataset(file_path):
    """Load dataset from CSV file"""
    print(f"Loading dataset from: {file_path}")
    
    # Try to read CSV
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

def prepare_data(df, sequence_length=10):
    """Prepare data for RNN training"""
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numeric columns found in dataset!")
        return None, None, None, None, None
    
    print(f"\nUsing {len(numeric_df.columns)} numeric columns: {numeric_df.columns.tolist()}")
    
    # Normalize data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(numeric_df)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_normalized) - sequence_length):
        X.append(data_normalized[i:i + sequence_length])
        y.append(data_normalized[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train (70%) and test (30%)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nData prepared:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Features per timestep: {X_train.shape[2]}")
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train, X_test, y_test, epochs=100):
    """Train the RNN model"""
    input_size = X_train.shape[2]
    hidden_size = 32
    output_size = y_train.shape[1]
    
    print(f"\nInitializing RNN:")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Output size: {output_size}")
    
    rnn = SimpleRNN(input_size, hidden_size, output_size, learning_rate=0.001)
    
    train_losses = []
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(X_train)):
            loss = rnn.train_step(X_train[i], y_train[i].reshape(-1, 1))
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(X_train)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    print("\nTraining complete!")
    return rnn, train_losses

def evaluate_model(rnn, X_test, y_test):
    """Evaluate model on test set"""
    print("\nEvaluating model on test set...")
    predictions = []
    
    for i in range(len(X_test)):
        pred, _ = rnn.forward(X_test[i])
        predictions.append(pred.flatten())
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\nTest Set Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    return predictions

def plot_results(y_test, predictions, train_losses):
    """Plot training loss and predictions"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot training loss
    axes[0].plot(train_losses)
    axes[0].set_title('Training Loss Over Time')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # Plot predictions vs actual (first feature only)
    num_samples = min(100, len(y_test))
    axes[1].plot(y_test[:num_samples, 0], label='Actual', linewidth=2)
    axes[1].plot(predictions[:num_samples, 0], label='Predicted', linewidth=2, alpha=0.7)
    axes[1].set_title('Predictions vs Actual (First Feature)')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Normalized Value')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('rnn_results.png')
    print("\nPlot saved as 'rnn_results.png'")
    plt.show()

def main():
    # ==== CONFIGURATION ====
    # Change this to your dataset path
    dataset_path = './Dataset/Midterm_53_group.csv'  # Modify this path
    sequence_length = 10  # Number of timesteps to look back
    epochs = 100  # Number of training epochs
    # =======================
    
    print("=" * 50)
    print("Network Traffic Predictor with RNN")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"\nError: Dataset file not found at '{dataset_path}'")
        print("Please update the 'dataset_path' variable with the correct path to your CSV file.")
        return
    
    # Load dataset
    df = load_dataset(dataset_path)
    if df is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, sequence_length)
    if X_train is None:
        return
    
    # Train model
    rnn, train_losses = train_model(X_train, y_train, X_test, y_test, epochs)
    
    # Evaluate model
    predictions = evaluate_model(rnn, X_test, y_test)
    
    # Plot results
    plot_results(y_test, predictions, train_losses)
    
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)

if __name__ == "__main__":
    main()