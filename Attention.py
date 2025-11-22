import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

class FeatureAttention(layers.Layer):
    """
    Feature-level attention mechanism
    Learns which features are most important for anomaly detection
    """
    def __init__(self, units=128, **kwargs):
        super(FeatureAttention, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        # Attention weights
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        # Context vector
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        
    def call(self, x):
        # Calculate attention scores
        uit = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        ait = tf.matmul(uit, tf.expand_dims(self.u, axis=-1))
        ait = tf.squeeze(ait, axis=-1)
        
        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(ait, axis=-1)
        
        # Apply attention weights
        weighted_input = x * tf.expand_dims(attention_weights, axis=-1)
        
        return weighted_input, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

def build_dense_with_attention(input_dim, output_size=2):
    """
    Dense Neural Network with Feature Attention
    The attention layer learns which features are most important
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Feature attention - learns importance of each feature
    attended_features, attention_weights = FeatureAttention(units=128)(inputs)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(attended_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(output_size, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def build_transformer_encoder(input_dim, output_size=2, num_heads=8, ff_dim=128):
    """
    Transformer Encoder for comparison
    Treats features as a sequence (even though they're not sequential)
    This is mostly for educational purposes - likely overkill for this task
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Reshape for transformer: (batch, 1, features)
    x = layers.Reshape((1, input_dim))(inputs)
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=input_dim // num_heads
    )(x, x)
    
    # Skip connection and normalization
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    
    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation='relu')(x)
    ff_output = layers.Dense(input_dim)(ff_output)
    
    # Skip connection and normalization
    x = layers.Add()([x, ff_output])
    x = layers.LayerNormalization()(x)
    
    # Flatten and classify
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(output_size, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def visualize_attention_weights(model, X_sample, feature_names=None):
    """
    Visualize which features the attention mechanism focuses on
    """
    import matplotlib.pyplot as plt
    
    # Create a model that outputs attention weights
    attention_layer = model.layers[1]  # FeatureAttention layer
    attention_model = models.Model(
        inputs=model.input,
        outputs=attention_layer.output[1]  # Get attention weights
    )
    
    # Get attention weights for sample
    attention_weights = attention_model.predict(X_sample[:100], verbose=0)
    avg_attention = np.mean(attention_weights, axis=0)
    
    # Plot top 20 most attended features
    top_indices = np.argsort(avg_attention)[-20:][::-1]
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(20), avg_attention[top_indices])
    
    if feature_names is not None:
        plt.yticks(range(20), [feature_names[i] for i in top_indices])
    else:
        plt.yticks(range(20), [f'Feature {i}' for i in top_indices])
    
    plt.xlabel('Average Attention Weight')
    plt.title('Top 20 Most Important Features (by Attention)')
    plt.tight_layout()
    plt.savefig('./Results/feature_attention.png', dpi=150)
    print("✓ Attention visualization saved to './Results/feature_attention.png'")
    plt.show()

# Example usage comparison
def compare_architectures():
    """
    Compare different architectures on your dataset
    """
    print("="*70)
    print("ARCHITECTURE COMPARISON")
    print("="*70)
    
    input_dim = 278  # Your dataset
    
    print("\n1. Dense Neural Network (Current - 99% accuracy)")
    print("   - Fast training")
    print("   - Simple and interpretable")
    print("   - Perfect for independent samples")
    
    print("\n2. Dense + Attention")
    print("   - Learns feature importance")
    print("   - Can visualize which features matter most")
    print("   - Might improve accuracy by 0.5-1%")
    print("   - Gives interpretability: 'Model focuses on packet size and duration'")
    model_attention = build_dense_with_attention(input_dim)
    model_attention.summary()
    
    print("\n3. Transformer Encoder")
    print("   - Powerful but overkill for this task")
    print("   - Slower training")
    print("   - Unlikely to beat dense network on independent samples")
    print("   - Better for: sequential data, language, time series")
    
    print("\n" + "="*70)
    print("RECOMMENDATION FOR YOUR PROJECT:")
    print("="*70)
    print("✅ Keep Dense NN - already 99% accurate!")
    print("💡 Add Attention layer for:")
    print("   - Feature importance visualization")
    print("   - Explainability in your report")
    print("   - Potentially 0.5-1% accuracy boost")
    print("❌ Skip Transformers - not suited for this problem")
    print("="*70)

if __name__ == "__main__":
    compare_architectures()