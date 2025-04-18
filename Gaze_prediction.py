import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io
from dataset import load 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import copy
from tensorflow.keras.regularizers import l2


sequence_length = 10
prediction_length = 1

dataset_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/merged_labeled_actions.mat'
mat_data = scipy.io.loadmat(dataset_path)
joints_data = mat_data['joints'] 

# Compute joint variances (how much each joint moves over time)
joint_variances = np.var(joints_data, axis=(1, 2)) 

# Compute average position of each joint (over time)
mean_positions = np.mean(joints_data, axis=2)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
joint_clusters = kmeans.fit_predict(mean_positions)

cluster_variances = np.zeros(n_clusters)
for cluster_id in range(n_clusters):
    cluster_variances[cluster_id] = np.mean(joint_variances[joint_clusters == cluster_id])

target_cluster = np.argmax(cluster_variances)
print(f"Most active cluster ID: {target_cluster}")


X_combined_, Y_object_past_, Y_object, Y_gaze_ = load(
    dataset_path, sequence_length, prediction_length, joint_clusters, target_cluster=target_cluster
)

if np.isscalar(Y_object_past_) or (isinstance(Y_object_past_, np.ndarray) and Y_object_past_.ndim == 0):
    num_samples = len(X_combined_)
    Y_object_past_ = np.zeros((num_samples, sequence_length), dtype=np.float32)


action_labels = np.load("/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/NewLabeledVideo/action_labels.npy")
Y_gaze_raw = action_labels[:, -2:]
print("Shape of raw gaze targets:", Y_gaze_raw.shape)
print("X_combined_ shape:", X_combined_.shape)
print("Y_object_past_ shape:", Y_object_past_.shape)
print("Y_gaze_ shape (raw):", Y_gaze_raw.shape)

if not (X_combined_.shape[0] == Y_object_past_.shape[0] == Y_gaze_raw.shape[0]):
    min_size = min(X_combined_.shape[0], Y_object_past_.shape[0], Y_gaze_raw.shape[0])
    print("Adjusting sizes to:", min_size)
    X_combined_ = X_combined_[:min_size]
    Y_object_past_ = Y_object_past_[:min_size]
    Y_gaze_raw = Y_gaze_raw[:min_size]

GAZE_MIN = np.min(Y_gaze_raw)
GAZE_MAX = np.max(Y_gaze_raw)
Y_gaze_norm = (Y_gaze_raw - GAZE_MIN) / (GAZE_MAX - GAZE_MIN)
print(f"GAZE_MIN: {GAZE_MIN}, GAZE_MAX: {GAZE_MAX}")

def add_noise(data, noise_factor=0.03):
    noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return np.clip(noisy_data, 0.0, 1.0)

X_combined_noisy = add_noise(np.array(X_combined_), noise_factor=0.02)


X_train, X_test, Y_past_train, Y_past_test, Y_gaze_train, Y_gaze_test = train_test_split(
    X_combined_noisy, Y_object_past_, Y_gaze_norm, test_size=0.2, random_state=42
)

X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_past_train_tensor = tf.convert_to_tensor(Y_past_train, dtype=tf.float32)
Y_gaze_train_tensor = tf.convert_to_tensor(Y_gaze_train, dtype=tf.float32)

X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_past_test_tensor = tf.convert_to_tensor(Y_past_test, dtype=tf.float32)
Y_gaze_test_tensor = tf.convert_to_tensor(Y_gaze_test, dtype=tf.float32)

train_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_train_tensor, Y_past_train_tensor), Y_gaze_train_tensor)
).batch(10)
test_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_test_tensor, Y_past_test_tensor), Y_gaze_test_tensor)
).batch(10)


class GazePredictionModelWithDecoder(tf.keras.Model):
    def __init__(self, parameters):
        super(GazePredictionModelWithDecoder, self).__init__()
        self.parameters = parameters
        self.hidden_state_size = parameters.get('hidden_state_size', 128)
        
        self.norm_layer = tf.keras.layers.LayerNormalization()

        self.feature_dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))
        self.feature_dense2 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4))
        self.feature_dense3 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-4))
        self.feature_dense4 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(1e-4))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_state_size, return_sequences=True, return_state=True,
                                   dropout=0.2, recurrent_dropout=0.2)
        )
        self.encoder_proj = tf.keras.layers.Dense(self.hidden_state_size)
        
        self.attention = tf.keras.layers.Attention()
        
        self.decoder = tf.keras.layers.LSTM(self.hidden_state_size, return_sequences=False, return_state=True,
                                            dropout=0.2, recurrent_dropout=0.2)
        
        self.state_proj_dense_h = tf.keras.layers.Dense(self.hidden_state_size)
        self.state_proj_dense_c = tf.keras.layers.Dense(self.hidden_state_size)
        
        self.output_layer = tf.keras.layers.Dense(2, activation='linear')
    
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
        
        decoder_output, _, _ = self.decoder(attn_output, initial_state=[state_h, state_c], training=training)
        
        gaze_pred = self.output_layer(decoder_output)  
        return gaze_pred


def train_step(model, optimizer, loss_fn, inputs, labels):
    with tf.GradientTape() as tape:
        gaze_pred = model(inputs, training=True) 
        loss = loss_fn(labels, gaze_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, gaze_pred

def compute_angular_error(y_true, y_pred):
    y_true_norm = tf.nn.l2_normalize(y_true, axis=1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=1)

    dot_product = tf.reduce_sum(y_true_norm * y_pred_norm, axis=1)
    dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)

    angles_rad = tf.acos(dot_product)
    angles_deg = angles_rad * (180.0 / np.pi)
    return angles_deg

def evaluate_model_with_angular(model, dataset, loss_fn, threshold=0.05):
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    angular_errors = []
    total_correct = 0.0
    total_count = 0.0

    for (X_batch, Y_past_batch), labels in dataset:
        preds = model((X_batch, Y_past_batch), training=False)
        mae_metric.update_state(labels, preds)

        distances = tf.norm(labels - preds, axis=1)
        correct = tf.reduce_sum(tf.cast(distances < threshold, tf.float32))
        total_correct += correct
        total_count += tf.cast(tf.shape(labels)[0], tf.float32)

        angles = compute_angular_error(labels, preds)
        angular_errors.append(angles)

    avg_angular_error = tf.reduce_mean(tf.concat(angular_errors, axis=0)).numpy()
    val_mae = mae_metric.result().numpy()
    val_accuracy = (total_correct / total_count).numpy()

    return val_mae, val_accuracy, avg_angular_error

def train_model_loop(model, train_dataset, test_dataset, epochs, optimizer, loss_fn, patience=5, threshold=0.05):
    best_val_mae = float('inf')
    best_val_accuracy = 0.0
    best_val_angular_error = float('inf')
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

        val_mae, val_accuracy, val_ang_err = evaluate_model_with_angular(model, test_dataset, loss_fn, threshold)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss.numpy():.4f}, "
              f"Train MAE: {train_mae_metric.result().numpy():.4f}, "
              f"Val MAE: {val_mae:.4f}, Val Accuracy: {val_accuracy:.4f}, "
              f"Angular Error: {val_ang_err:.2f}°")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_accuracy = val_accuracy
            best_val_angular_error = val_ang_err
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs.\n"
                  f"Best Val MAE: {best_val_mae:.4f}, "
                  f"Best Val Accuracy: {best_val_accuracy:.4f}, "
                  f"Best Angular Error: {best_val_angular_error:.2f}°")
            break

vocab_size = 11 
parameters = {'sequence_length': sequence_length, 'vocab_size': vocab_size, 'hidden_state_size': 100}
model = GazePredictionModelWithDecoder(parameters)

feature_dim = X_train_tensor.shape[-1]
dummy_input = (
    tf.random.uniform((1, sequence_length, feature_dim)),
    tf.random.uniform((1, sequence_length), maxval=10, dtype=tf.int32)
)
_ = model(dummy_input, training=False)

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-4,
    first_decay_steps=500,
    t_mul=2.0,
    m_mul=0.9
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.MeanAbsoluteError()

train_model_loop(model, train_dataset, test_dataset, epochs=20, optimizer=optimizer,
                 loss_fn=loss_fn, patience=3, threshold=0.05)