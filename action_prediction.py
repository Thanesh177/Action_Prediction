import tensorflow as tf
import numpy as np
from dataset import load
import scipy.io
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

# Dataset parameters
sequence_length = 10
prediction_length = 1

# Load the dataset
dataset_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/merged_labeled_actions.mat'
mat_data = scipy.io.loadmat(dataset_path)
joints_data = mat_data['joints']  # shape: (n_joints, n_frames, 3)

# Step 1: Compute joint variances (how much each joint moves over time)
joint_variances = np.var(joints_data, axis=(1, 2))  # shape: (n_joints,)

# Step 2: Compute average position of each joint (over time)
mean_positions = np.mean(joints_data, axis=2)  # shape: (n_joints, n_frames)

# Step 3: Apply K-means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
joint_clusters = kmeans.fit_predict(mean_positions)

# Step 4: Aggregate variance by cluster
cluster_variances = np.zeros(n_clusters)
for cluster_id in range(n_clusters):
    cluster_variances[cluster_id] = np.mean(joint_variances[joint_clusters == cluster_id])

# Step 5: Identify the most "active" cluster (the one with highest average joint variance)
target_cluster = np.argmax(cluster_variances)
print(f"Most active cluster ID: {target_cluster}")

# Load processed dataset
X_combined_, Y_object_past_, Y_object_, sequence_lengths = load(
    dataset_path, sequence_length, prediction_length, joint_clusters, target_cluster
)

# Add Gaussian Noise for Data Augmentation
def add_noise(data, noise_factor=0.05):
    noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return np.clip(noisy_data, 0.0, 1.0)

X_combined_noisy = add_noise(np.array(X_combined_))

# Train-Test Split
X_train, X_test, Y_past_train, Y_past_test, Y_train, Y_test = train_test_split(
    X_combined_noisy, Y_object_past_, Y_object_, test_size=0.2, random_state=42
)

# Convert to TensorFlow tensors (avoid RaggedTensor)
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_past_train_tensor = tf.convert_to_tensor(Y_past_train, dtype=tf.float32)
Y_train_tensor = tf.convert_to_tensor(Y_train, dtype=tf.int32)

X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_past_test_tensor = tf.convert_to_tensor(Y_past_test, dtype=tf.float32)
Y_test_tensor = tf.convert_to_tensor(Y_test, dtype=tf.int32)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_train_tensor, Y_past_train_tensor), Y_train_tensor)
).batch(30)

test_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_test_tensor, Y_past_test_tensor), Y_test_tensor)
).batch(30)

# Model Definition
class ActionPredictionModel(tf.keras.Model):
    def __init__(self, parameters):
        super(ActionPredictionModel, self).__init__()
        self.parameters = parameters
        self.hidden_state_size = parameters.get('hidden_state_size', 25)

        self.norm_layer = tf.keras.layers.LayerNormalization()
        self.feature_dense1 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-4))
        self.feature_dense2 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(1e-4))
        self.feature_dense3 = tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2(1e-4))

        self.dropout = tf.keras.layers.Dropout(0.3)
        self.batch_norm = tf.keras.layers.BatchNormalization()

        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_state_size, return_sequences=True, return_state=True,
                                   dropout=0.2, recurrent_dropout=0.2)
        )

        self.attention = tf.keras.layers.Attention()

        self.decoder = tf.keras.layers.LSTM(self.hidden_state_size, return_sequences=True, return_state=True,
                                            dropout=0.2, recurrent_dropout=0.2)
        self.output_layer = tf.keras.layers.Dense(self.parameters['vocab_size'], activation='softmax')

    def call(self, inputs, training=False):
        X_combined, Y_past = inputs
        X_combined = self.norm_layer(X_combined)

        combined_embedding = self.feature_dense1(X_combined)
        combined_embedding = self.dropout(combined_embedding, training=training)
        combined_embedding = self.batch_norm(combined_embedding, training=training)
        combined_embedding = self.feature_dense2(combined_embedding)
        combined_embedding = self.dropout(combined_embedding, training=training)
        combined_embedding = self.feature_dense3(combined_embedding)

        # Concatenate the combined features with Y_past (expanded to add a feature dimension)
        input_sequence = tf.concat([
            combined_embedding,
            tf.expand_dims(tf.cast(Y_past, tf.float32), axis=-1)
        ], axis=2)

        # Encoder: Bidirectional LSTM
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder(input_sequence)
        
        # Combine forward and backward states
        state_h = tf.concat([forward_h, backward_h], axis=-1)  # (batch_size, hidden_state_size * 2)
        state_c = tf.concat([forward_c, backward_c], axis=-1)  # (batch_size, hidden_state_size * 2)

        # Reduce states to match decoder's expected hidden size
        state_h = tf.keras.layers.Dense(self.hidden_state_size)(state_h)  # (batch_size, hidden_state_size)
        state_c = tf.keras.layers.Dense(self.hidden_state_size)(state_c)  # (batch_size, hidden_state_size)

        # Build query for attention. We want its last dimension to match that of encoder_outputs.
        query = tf.keras.layers.Dense(self.hidden_state_size * 2)(state_h)  
        query = tf.expand_dims(query, axis=1)  # (batch_size, 1, hidden_state_size * 2)

        # Apply attention: Query and encoder_outputs (as key/value) must have matching last dimensions.
        attention_output = self.attention([query, encoder_outputs])

        # Decoder: LSTM using the attention output and initial state from the encoder.
        decoder_outputs, _, _ = self.decoder(attention_output, initial_state=[state_h, state_c])
        predictions = self.output_layer(decoder_outputs)
        return predictions

# Training Utilities
def train_step(model, optimizer, loss_fn, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        # Remove the extra sequence dimension if needed (assumes predictions shape is (batch, 1, vocab_size))
        predictions = tf.squeeze(predictions, axis=1)
        labels = tf.one_hot(tf.squeeze(labels, axis=1), depth=model.parameters['vocab_size'])
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

def evaluate_model(model, dataset):
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
    for (X_combined_batch, Y_past_batch), Y_batch in dataset:
        predictions = model((X_combined_batch, Y_past_batch), training=False)
        predictions = tf.squeeze(predictions, axis=1)
        labels = tf.one_hot(tf.squeeze(Y_batch, axis=1), depth=model.parameters['vocab_size'])
        accuracy_metric.update_state(labels, predictions)
    return accuracy_metric.result().numpy()

def train_model(model, train_dataset, test_dataset, epochs, optimizer, loss_fn, patience=5):
    best_val_accuracy = 0.0
    no_improvement_epochs = 0
    for epoch in range(epochs):
        epoch_loss = 0
        accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
        for (X_combined_batch, Y_past_batch), Y_batch in train_dataset:
            loss, predictions = train_step(model, optimizer, loss_fn, (X_combined_batch, Y_past_batch), Y_batch)
            epoch_loss += loss
            accuracy_metric.update_state(tf.one_hot(tf.squeeze(Y_batch, axis=1), depth=model.parameters['vocab_size']), predictions)
        val_accuracy = evaluate_model(model, test_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss.numpy()}, Accuracy: {accuracy_metric.result().numpy()}, Validation Accuracy: {val_accuracy}")



# Prepare model parameters and instantiate the model
vocab_size = 6
parameters = {'sequence_length': sequence_length, 'vocab_size': vocab_size, 'hidden_state_size': 35}
model = ActionPredictionModel(parameters)

# Force building the model by running a dummy input through it
feature_dim = X_train_tensor.shape[-1]
dummy_input = (
    tf.random.uniform((1, sequence_length, feature_dim)),
    tf.random.uniform((1, sequence_length), maxval=vocab_size, dtype=tf.int32)
)
_ = model(dummy_input, training=False)

# Set up optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Train the model
train_model(model, train_dataset, test_dataset, epochs=20, optimizer=optimizer, loss_fn=loss_fn)