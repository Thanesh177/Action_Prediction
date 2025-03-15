import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io
from dataset import load  # Assumes load() returns: X_combined_, Y_object_past_, Y_action_, Y_gaze_
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import copy
from tensorflow.keras.regularizers import l2

# ============================
# Part 1. Data Loading and Preprocessing
# ============================
dataset_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/merged_labeled_actions.mat'
sequence_length = 4
prediction_length = 1

# --- (Optional) Compute clustering on joint data ---
mat_data = scipy.io.loadmat(dataset_path)
joints_data = mat_data['joints']
joint_variances = np.var(joints_data, axis=(1, 2))
mean_positions = np.mean(joints_data, axis=2)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
joint_clusters = kmeans.fit_predict(mean_positions)
target_cluster = 4

# --- Load processed dataset for inputs ---
# We load X_combined_, Y_object_past_, and Y_action_ (which we drop later)
X_combined_, Y_object_past_, Y_action_, Y_gaze_ = load(
    dataset_path, sequence_length, prediction_length, joint_clusters, target_cluster=target_cluster
)
# Convert action labels from 1-indexed to 0-indexed (not used in this model)
Y_action_ = Y_action_ - 1

if np.isscalar(Y_object_past_) or (isinstance(Y_object_past_, np.ndarray) and Y_object_past_.ndim == 0):
    num_samples = len(X_combined_)
    Y_object_past_ = np.zeros((num_samples, sequence_length), dtype=np.float32)

# --- Load raw gaze coordinates from the action_labels file ---
# We assume the last two columns of action_labels.npy contain the raw gaze coordinates.
action_labels = np.load("/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/NewLabeledVideo/action_labels.npy")
Y_gaze_raw = action_labels[:, -2:]
print("Shape of raw gaze targets:", Y_gaze_raw.shape)
print("X_combined_ shape:", X_combined_.shape)
print("Y_object_past_ shape:", Y_object_past_.shape)
print("Y_gaze_ shape (raw):", Y_gaze_raw.shape)

# Ensure all arrays have the same number of samples.
if not (X_combined_.shape[0] == Y_object_past_.shape[0] == Y_gaze_raw.shape[0]):
    min_size = min(X_combined_.shape[0], Y_object_past_.shape[0], Y_gaze_raw.shape[0])
    print("Adjusting sizes to:", min_size)
    X_combined_ = X_combined_[:min_size]
    Y_object_past_ = Y_object_past_[:min_size]
    Y_gaze_raw = Y_gaze_raw[:min_size]

# --- Normalize gaze targets using min–max normalization ---
GAZE_MIN = np.min(Y_gaze_raw)
GAZE_MAX = np.max(Y_gaze_raw)
Y_gaze_norm = (Y_gaze_raw - GAZE_MIN) / (GAZE_MAX - GAZE_MIN)
print(f"GAZE_MIN: {GAZE_MIN}, GAZE_MAX: {GAZE_MAX}")

# --- Data Augmentation: Add Gaussian Noise to inputs ---
def add_noise(data, noise_factor=0.03):
    noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return np.clip(noisy_data, 0.0, 1.0)

X_combined_noisy = add_noise(np.array(X_combined_), noise_factor=0.02)

# --- Train-Test Split ---
# We use X_combined (augmented) and Y_object_past_ (as Y_past) as inputs, and normalized gaze coordinates as targets.
X_train, X_test, Y_past_train, Y_past_test, Y_gaze_train, Y_gaze_test = train_test_split(
    X_combined_noisy, Y_object_past_, Y_gaze_norm, test_size=0.2, random_state=42
)

# Convert arrays to TF tensors.
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_past_train_tensor = tf.convert_to_tensor(Y_past_train, dtype=tf.float32)
Y_gaze_train_tensor = tf.convert_to_tensor(Y_gaze_train, dtype=tf.float32)

X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_past_test_tensor = tf.convert_to_tensor(Y_past_test, dtype=tf.float32)
Y_gaze_test_tensor = tf.convert_to_tensor(Y_gaze_test, dtype=tf.float32)

# Create datasets where inputs are (X_combined, Y_past) and target is normalized Y_gaze.
train_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_train_tensor, Y_past_train_tensor), Y_gaze_train_tensor)
).batch(17)
test_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_test_tensor, Y_past_test_tensor), Y_gaze_test_tensor)
).batch(17)

# ============================
# Part 2. Model Definition (Dense Layers Only, Without Action Label, Two Outputs)
class GazePredictionModelWithDecoder(tf.keras.Model):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.hidden_state_size = parameters.get('hidden_state_size', 50)
        
        # Layers
        self.norm_layer = tf.keras.layers.LayerNormalization()

        # Dense layers
        self.feature_dense1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4))
        self.feature_dense2 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-4))
        self.feature_dense3 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(1e-4))
        self.feature_dense4 = tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2(1e-4))
        
        # Batch Norm
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(0.3)
        
        # Encoder
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_state_size, return_sequences=True, return_state=True, dropout=0.2, recurrent_dropout=0.2)
        )
        
        # Attention & Decoder
        self.attention = tf.keras.layers.Attention()
        self.decoder = tf.keras.layers.LSTM(self.hidden_state_size, return_sequences=False, return_state=True, dropout=0.2, recurrent_dropout=0.2)
        
        # State projection layers (new!)
        self.state_proj_dense_h = tf.keras.layers.Dense(self.hidden_state_size)
        self.state_proj_dense_c = tf.keras.layers.Dense(self.hidden_state_size)

        # Project encoder outputs from 2*hidden_state_size to hidden_state_size for attention
        self.encoder_proj = tf.keras.layers.Dense(self.hidden_state_size)

        # Output
        self.output_layer = tf.keras.layers.Dense(2, activation='sigmoid')
    
    def call(self, inputs, training=False):
        X_combined, Y_past = inputs
        
        X_combined = self.norm_layer(X_combined)
        x = self.feature_dense1(X_combined)
        x = self.batch_norm1(x, training=training)
        x = self.feature_dense2(x)
        x = self.dropout(x, training=training)
        x = self.feature_dense3(x)
        x = self.dropout(x, training=training)
        x = self.feature_dense4(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout(x, training=training)
        
        Y_past = tf.expand_dims(tf.cast(Y_past, tf.float32), axis=-1)
        input_sequence = tf.concat([x, Y_past], axis=2)
        
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder(input_sequence, training=training)
        encoder_outputs = self.encoder_proj(encoder_outputs)
        
        state_h = self.state_proj_dense_h(tf.concat([forward_h, backward_h], axis=-1))
        state_c = self.state_proj_dense_c(tf.concat([forward_c, backward_c], axis=-1))
        
        query = tf.expand_dims(state_h, axis=1)
        attn_output = self.attention([query, encoder_outputs])
        
        decoder_output, dec_state_h, dec_state_c = self.decoder(attn_output, initial_state=[state_h, state_c], training=training)
        gaze_pred = self.output_layer(decoder_output)
        return gaze_pred, dec_state_h, dec_state_c

    def build(self, input_shape):
        super(GazePredictionModelWithDecoder, self).build(input_shape)

# ============================
# Part 3. Training Utilities and Training Loop with Early Stopping
# ============================
def train_step(model, optimizer, loss_fn, inputs, labels):
    with tf.GradientTape() as tape:
        gaze_pred, _, _ = model(inputs, training=True)
        loss = loss_fn(labels, gaze_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, gaze_pred

def evaluate_model(model, dataset, loss_fn, threshold=0.05):
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    total_correct = 0.0
    total_count = 0.0
    for (X_combined_batch, Y_past_batch), labels in dataset:
        gaze_pred, _, _ = model((X_combined_batch, Y_past_batch), training=False)
        true_gaze = labels
        mae_metric.update_state(true_gaze, gaze_pred)
        distances = tf.norm(true_gaze - gaze_pred, axis=1)
        correct = tf.reduce_sum(tf.cast(distances < threshold, tf.float32))
        total_correct += correct
        total_count += tf.cast(tf.shape(true_gaze)[0], tf.float32)
    val_mae = mae_metric.result().numpy()
    val_accuracy = (total_correct / total_count).numpy()
    return val_mae, val_accuracy

def train_model_loop(model, train_dataset, test_dataset, epochs, optimizer, loss_fn, patience=5, threshold=0.05):
    best_val_mae = float('inf')
    best_val_accuracy = 0.0
    no_improvement_epochs = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        train_mae_metric = tf.keras.metrics.MeanAbsoluteError()
        for (X_combined_batch, Y_past_batch), labels in train_dataset:
            loss, gaze_pred = train_step(model, optimizer, loss_fn,
                                          (X_combined_batch, Y_past_batch),
                                          labels)
            epoch_loss += loss
            train_mae_metric.update_state(labels, gaze_pred)
        val_mae, val_accuracy = evaluate_model(model, test_dataset, loss_fn, threshold)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss.numpy()}, Train MAE: {train_mae_metric.result().numpy()}, Val MAE: {val_mae}, Val Accuracy: {val_accuracy}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_accuracy = val_accuracy
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        if no_improvement_epochs >= patience:
            print(f"Early stopping after {epoch+1} epochs. Best Val MAE: {best_val_mae}, Best Val Accuracy: {best_val_accuracy}")
            break

vocab_size = len(np.unique(Y_action_))  # Still defined, even if not used.
parameters = {'sequence_length': sequence_length, 'vocab_size': vocab_size, 'hidden_state_size': 100}
model = GazePredictionModelWithDecoder(parameters)

# Build model with dummy input to initialize weights.
feature_dim = X_train_tensor.shape[-1]
dummy_input = (
    tf.random.uniform((1, sequence_length, feature_dim)),
    tf.random.uniform((1, sequence_length), maxval=10, dtype=tf.int32)
)
_ = model(dummy_input, training=False)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.523,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.MeanAbsoluteError()

train_model_loop(model, train_dataset, test_dataset, epochs=2, optimizer=optimizer,
                 loss_fn=loss_fn, patience=3, threshold=0.05)

