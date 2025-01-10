import tensorflow as tf
import numpy as np
from dataset import load  # Adjust the import path based on your project structure
import scipy.io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Dataset parameters
dataset_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/merged_labeled_actions.mat'
sequence_length = 10
prediction_length = 5

mat_data = scipy.io.loadmat(dataset_path)

    # Extract the joint data
joints_data = mat_data['joints']

    # Compute the variance across the temporal and sample dimensions for each joint feature
joint_variances = np.var(joints_data, axis=(1, 2))

    # Compute the mean joint positions over time to simplify analysis
mean_positions = np.mean(joints_data, axis=2)

    # Apply K-means clustering to the mean positions of the joints
n_clusters = 5  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
joint_clusters = kmeans.fit_predict(mean_positions)
target_cluster = 4  # Focus on Cluster 4

# Load the dataset
X_combined_, Y_object_past_, Y_object_, sequence_lengths = load(
    dataset_path, sequence_length, prediction_length, joint_clusters, target_cluster=4
)

# Convert data to TensorFlow tensors
X_combined_tensor = tf.ragged.constant(X_combined_)
Y_object_past_tensor = tf.ragged.constant(Y_object_past_)
Y_object_tensor = tf.ragged.constant(Y_object_)

print("Y_object_ sample:", Y_object_[:1])
print("Y_object_ shape:", len(Y_object_))

# Define dataset
dataset = tf.data.Dataset.from_tensor_slices((
    X_combined_tensor, Y_object_past_tensor, Y_object_tensor
)).batch(32)


class ActionPredictionModel(tf.keras.Model):
    def __init__(self, parameters):
        super(ActionPredictionModel, self).__init__()
        self.parameters = parameters
        self.hidden_state_size = parameters.get('hidden_state_size', 128)

        # Normalization Layer
        self.norm_layer = tf.keras.layers.LayerNormalization()

        # Combined Feature Processing
        self.feature_dense1 = tf.keras.layers.Dense(50, activation='relu')
        self.feature_dense2 = tf.keras.layers.Dense(20, activation='relu')
        self.feature_dense3 = tf.keras.layers.Dense(10, activation='relu')

        # Context Encoder with Attention
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.hidden_state_size, return_sequences=True, return_state=True
            )
        )
        self.attention = tf.keras.layers.Attention()

        # Decoder
        self.decoder = tf.keras.layers.LSTM(
            self.hidden_state_size, return_sequences=True, return_state=True
        )

        # Output Layer
        self.output_layer = tf.keras.layers.Dense(
            self.parameters['vocab_size'], activation='softmax'
        )

    def call(self, inputs, training=False):
        X_combined, Y_past = inputs

        # Normalize combined features
        X_combined = self.norm_layer(X_combined)

        # Process features
        combined_embedding = self.feature_dense1(X_combined)
        combined_embedding = self.feature_dense2(combined_embedding)
        combined_embedding = self.feature_dense3(combined_embedding)

        # Combine with past labels
        input_sequence = tf.concat([
            combined_embedding,
            tf.expand_dims(tf.cast(Y_past, tf.float32), axis=-1)
        ], axis=2)

        # Encode context
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder(input_sequence)

        # Combine forward and backward states
        state_h = tf.concat([forward_h, backward_h], axis=-1)
        state_c = tf.concat([forward_c, backward_c], axis=-1)

        # Project combined states to match decoder dimensions
        state_h = tf.keras.layers.Dense(self.hidden_state_size)(state_h)
        state_c = tf.keras.layers.Dense(self.hidden_state_size)(state_c)

        # Apply attention
        attention_output = self.attention([encoder_outputs, encoder_outputs])

        # Decode predictions
        decoder_outputs, _, _ = self.decoder(attention_output, initial_state=[state_h, state_c])

        # Generate output
        predictions = self.output_layer(decoder_outputs)
        return predictions


# Training utilities
def train_step(model, optimizer, loss_fn, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        sliced_predictions = predictions[:, -labels.shape[1]:, :]  # Slice predictions for the last prediction_length
        loss = loss_fn(tf.one_hot(labels, depth=model.parameters['vocab_size']), sliced_predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, sliced_predictions

def evaluate_model(model, dataset):
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    for X_combined_batch, Y_past_batch, Y_batch in dataset:
        inputs = (
            X_combined_batch.to_tensor(),
            Y_past_batch.to_tensor(),
        )
        labels = tf.one_hot(Y_batch.to_tensor(), depth=model.parameters['vocab_size'])
        predictions = model(inputs, training=False)
        sliced_predictions = predictions[:, -labels.shape[1]:, :]  # Slice predictions for the last prediction_length

        # Update accuracy metric
        accuracy_metric.update_state(labels, sliced_predictions)

    return accuracy_metric.result().numpy()

def train_model(model, dataset, epochs, optimizer, loss_fn, patience=3):
    train_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    best_val_accuracy = 0.0
    no_improvement_epochs = 0

    for epoch in range(epochs):
        epoch_loss = 0
        accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

        # Training loop
        for X_combined_batch, Y_past_batch, Y_batch in dataset:
            inputs = (
                X_combined_batch.to_tensor(),
                Y_past_batch.to_tensor(),
            )
            labels = Y_batch.to_tensor()

            # Perform a training step
            loss, sliced_predictions = train_step(model, optimizer, loss_fn, inputs, labels)
            epoch_loss += loss

            # Update metrics
            accuracy_metric.update_state(tf.one_hot(labels, depth=model.parameters['vocab_size']), sliced_predictions)

        # Store training metrics
        train_loss_history.append(epoch_loss.numpy())
        train_accuracy_history.append(accuracy_metric.result().numpy())

        # Validation accuracy
        val_accuracy = evaluate_model(model, dataset)
        val_accuracy_history.append(val_accuracy)

        # Print metrics
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss.numpy()}, Accuracy: {accuracy_metric.result().numpy()}, Validation Accuracy: {val_accuracy}")

        # Check for early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_epochs = 0  # Reset counter if validation accuracy improves
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs. Best Validation Accuracy: {best_val_accuracy}")
            break

    return train_loss_history, train_accuracy_history, val_accuracy_history

if len(Y_object_) == 0:
    raise ValueError("Y_object_ is empty after dataset processing. Check the input data or processing pipeline.")

parameters = {
    'sequence_length': sequence_length,
    'vocab_size': 10,
    'hidden_state_size': 128,
}

# Initialize the model, optimizer, and loss function
model = ActionPredictionModel(parameters)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.95
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.CategoricalCrossentropy()


# Train the model
epochs = 10
train_loss_history, train_accuracy_history, val_accuracy_history = train_model(
    model, dataset, epochs, optimizer, loss_fn
)

# Evaluate model accuracy, precision, and recall
accuracy= evaluate_model(model, dataset)
print(f"Final Training Accuracy: {accuracy}")

def plot_training_history(loss_history, train_accuracy_history, val_accuracy_history):
    epochs = range(1, len(loss_history) + 1)

    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, label='Training Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy_history, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_accuracy_history, label='Validation Accuracy', marker='x', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()


    plt.tight_layout()
    plt.show()

plot_training_history(train_loss_history, train_accuracy_history, val_accuracy_history)