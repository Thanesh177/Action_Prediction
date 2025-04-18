import tensorflow as tf
import numpy as np
from dataset import load
import scipy.io
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

sequence_length = 10
prediction_length = 1

dataset_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/merged_labeled_actions.mat'
mat_data = scipy.io.loadmat(dataset_path)
joints_data = mat_data['joints'] 

# Compute joint variances (how much each joint moves over time)
joint_variances = np.var(joints_data, axis=(1, 2)) 

#  Compute average position of each joint (over time)
mean_positions = np.mean(joints_data, axis=2)  

n_clusters = 1
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
joint_clusters = kmeans.fit_predict(mean_positions)

cluster_variances = np.zeros(n_clusters)
for cluster_id in range(n_clusters):
    cluster_variances[cluster_id] = np.mean(joint_variances[joint_clusters == cluster_id])

target_cluster = np.argmax(cluster_variances)
print(f"Most active cluster ID: {target_cluster}")

X_combined_, Y_object_past_, Y_object_, sequence_lengths = load(
    dataset_path, sequence_length, prediction_length, joint_clusters, target_cluster
)

def add_noise(data, noise_factor=0.05):
    noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return np.clip(noisy_data, 0.0, 1.0)

X_combined_noisy = add_noise(np.array(X_combined_))

X_train, X_test, Y_past_train, Y_past_test, Y_train, Y_test = train_test_split(
    X_combined_noisy, Y_object_past_, Y_object_, test_size=0.2, random_state=42
)

X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_past_train_tensor = tf.convert_to_tensor(Y_past_train, dtype=tf.float32)
Y_train_tensor = tf.convert_to_tensor(Y_train, dtype=tf.int32)

X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_past_test_tensor = tf.convert_to_tensor(Y_past_test, dtype=tf.float32)
Y_test_tensor = tf.convert_to_tensor(Y_test, dtype=tf.int32)

train_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_train_tensor, Y_past_train_tensor), Y_train_tensor)
).batch(30)

test_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_test_tensor, Y_past_test_tensor), Y_test_tensor)
).batch(30)

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

        input_sequence = tf.concat([
            combined_embedding,
            tf.expand_dims(tf.cast(Y_past, tf.float32), axis=-1)
        ], axis=2)

        encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder(input_sequence)
        
        state_h = tf.concat([forward_h, backward_h], axis=-1)  
        state_c = tf.concat([forward_c, backward_c], axis=-1)  

        state_h = tf.keras.layers.Dense(self.hidden_state_size)(state_h) 
        state_c = tf.keras.layers.Dense(self.hidden_state_size)(state_c)  

        query = tf.keras.layers.Dense(self.hidden_state_size * 2)(state_h)  
        query = tf.expand_dims(query, axis=1) 

        attention_output = self.attention([query, encoder_outputs])

        decoder_outputs, _, _ = self.decoder(attention_output, initial_state=[state_h, state_c])
        predictions = self.output_layer(decoder_outputs)
        return predictions

def train_step(model, optimizer, loss_fn, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
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
    best_weights = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

        for (X_combined_batch, Y_past_batch), Y_batch in train_dataset:
            loss, predictions = train_step(model, optimizer, loss_fn, (X_combined_batch, Y_past_batch), Y_batch)
            epoch_loss += loss.numpy()
            accuracy_metric.update_state(tf.one_hot(tf.squeeze(Y_batch, axis=1), depth=model.parameters['vocab_size']), predictions)

        val_accuracy = evaluate_model(model, test_dataset)
        train_accuracy = accuracy_metric.result().numpy()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, "
              f"Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Early stopping logic
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_epochs = 0
            best_weights = model.get_weights()  # Save best weights
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

    # Restore best weights if early stopping occurred
    if best_weights is not None:
        model.set_weights(best_weights)


vocab_size = 11
parameters = {'sequence_length': sequence_length, 'vocab_size': vocab_size, 'hidden_state_size': 35}
model = ActionPredictionModel(parameters)

feature_dim = X_train_tensor.shape[-1]
dummy_input = (
    tf.random.uniform((1, sequence_length, feature_dim)),
    tf.random.uniform((1, sequence_length), maxval=vocab_size, dtype=tf.int32)
)
_ = model(dummy_input, training=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

train_model(model, train_dataset, test_dataset, epochs=20, optimizer=optimizer, loss_fn=loss_fn,  patience=4)