import tensorflow as tf
import numpy as np
from dataset import load
import scipy.io
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

# Dataset parameters
dataset_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/merged_labeled_actions.mat'
sequence_length = 10
prediction_length = 1

# Load the dataset
mat_data = scipy.io.loadmat(dataset_path)
joints_data = mat_data['joints']
joint_variances = np.var(joints_data, axis=(1, 2))
mean_positions = np.mean(joints_data, axis=2)

# Apply K-means clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
joint_clusters = kmeans.fit_predict(mean_positions)
target_cluster = 4

# Load processed dataset
X_combined_, Y_object_past_, Y_object_, sequence_lengths = load(
    dataset_path, sequence_length, prediction_length, joint_clusters, target_cluster=4
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
).shuffle(1000).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_test_tensor, Y_past_test_tensor), Y_test_tensor)
).batch(32)

# Model Definition
class ActionPredictionModel(tf.keras.Model):
    def __init__(self, parameters):
        super(ActionPredictionModel, self).__init__()
        self.parameters = parameters
        self.hidden_state_size = parameters.get('hidden_state_size', 20)
        self.vocab_size = parameters.get('vocab_size', 6)

        # Normalization and feature extraction layers
        self.norm_layer = tf.keras.layers.LayerNormalization()
        self.feature_dense1 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-4))
        self.feature_dense2 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(1e-4))
        self.feature_dense3 = tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2(1e-4))
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.batch_norm = tf.keras.layers.BatchNormalization()

        # Encoder: Bidirectional LSTM with attention
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_state_size, return_sequences=True, return_state=True,
                                   dropout=0.2, recurrent_dropout=0.2)
        )
        self.attention = tf.keras.layers.Attention()
        self.state_h_dense = tf.keras.layers.Dense(self.hidden_state_size)
        self.state_c_dense = tf.keras.layers.Dense(self.hidden_state_size)
        self.query_dense = tf.keras.layers.Dense(self.hidden_state_size * 2)

        # Decoder and Output
        self.decoder = tf.keras.layers.LSTM(self.hidden_state_size, return_sequences=True, return_state=True,
                                            dropout=0.2, recurrent_dropout=0.2)
        self.output_layer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def build(self, input_shape):
        # input_shape is a tuple: (X_seq_shape, Y_past_shape)
        # Ensure that all sublayers are built with appropriate input shapes.
        feature_shape = input_shape[0]  # e.g., (batch_size, sequence_length, feature_dim)
        self.norm_layer.build(feature_shape)
        self.feature_dense1.build(feature_shape)
        # Assuming the output of feature_dense1 has last dim = 32.
        self.feature_dense2.build((feature_shape[0], feature_shape[1], 32))
        self.feature_dense3.build((feature_shape[0], feature_shape[1], 16))
        # Build remaining layers by passing dummy data if needed.
        super(ActionPredictionModel, self).build(input_shape)

def call(self, inputs, training=False):
    # Expect inputs as a tuple: (X_sequence, Y_past)
    X_seq, Y_past = inputs
    # Cast X_seq to float16 as desired (global policy)
    X_seq = tf.cast(X_seq, dtype=tf.float16)
    X_norm = self.norm_layer(X_seq)
    x = self.feature_dense1(X_norm)
    x = self.batch_norm(x, training=training)
    x = self.dropout(x, training=training)
    x = self.feature_dense2(x)
    x = self.dropout(x, training=training)
    x = self.feature_dense3(x)
    
    # BatchNormalization often returns float32 values even with mixed precision.
    # So cast Y_past to float32 (instead of float16) to match x's dtype.
    Y_past_cast = tf.expand_dims(tf.cast(Y_past, tf.float32), axis=-1)
    input_sequence = tf.concat([x, Y_past_cast], axis=2)

    # Encoder: process the concatenated sequence
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder(input_sequence)
    state_h = tf.concat([forward_h, backward_h], axis=-1)
    state_c = tf.concat([forward_c, backward_c], axis=-1)

    # Reduce states to match decoder's hidden state size.
    state_h = self.state_h_dense(state_h)
    state_c = self.state_c_dense(state_c)

    # Build attention query from state_h.
    query = self.query_dense(state_h)
    query = tf.expand_dims(query, axis=1)
    attention_output = self.attention([query, encoder_outputs])
    attention_output = tf.squeeze(attention_output, axis=1)

    # Decoder: generate predictions.
    decoder_outputs, _, _ = self.decoder(tf.expand_dims(attention_output, axis=1),
                                         initial_state=[state_h, state_c])
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

        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        if no_improvement_epochs >= patience:
            print(f"Early stopping after {epoch+1} epochs. Best Validation Accuracy: {best_val_accuracy}")
            break
        

# Prepare model parameters and instantiate the model
vocab_size = len(np.unique(Y_object_))
parameters = {'sequence_length': sequence_length, 'vocab_size': vocab_size, 'hidden_state_size': 30}
model = ActionPredictionModel(parameters)

# Force building the model by running a dummy input through it
feature_dim = X_train_tensor.shape[-1]
dummy_input = (
    tf.random.uniform((1, sequence_length, feature_dim)),
    tf.random.uniform((1, sequence_length), maxval=vocab_size, dtype=tf.int32)
)
_ = model(dummy_input, training=False)

# Set up optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Train the model
train_model(model, train_dataset, test_dataset, epochs=20, optimizer=optimizer, loss_fn=loss_fn)