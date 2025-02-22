import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

# Enable XLA for speed and set mixed precision for performance
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ------------------------------------------------------
# 1. Data Loading and Preprocessing
# ------------------------------------------------------
file_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/prediction/Fully_Cleaned_Dataset.csv'
data = pd.read_csv(file_path, header=None)
data.columns = ['Head_Vel', 'HeadVector_X', 'HeadVector_Y', 'HeadVector_Z', 'Label']

# For quicker experimentation, sample 50% of the dataset
data = data.sample(frac=0.4, random_state=42).reset_index(drop=True)

# Extract features and labels
X = data[['Head_Vel', 'HeadVector_X', 'HeadVector_Y', 'HeadVector_Z']].values
y = data['Label'].values.astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------
# 2. Sequence Creation Function (with X_past and Y_past)
# ------------------------------------------------------
def create_reference(X_combined, Y_object, sequence_length, prediction_length):
    """
    Creates sliding-window sequences.
    
    For each full sequence (assumed here to be a continuous stream), this function extracts:
      - X_past: A window of past feature values (length = sequence_length)
      - Y_object_past: The corresponding past labels (length = sequence_length)
      - Y_object_future: The future label(s) (length = prediction_length)
    
    Parameters:
      X_combined: list/array of feature sequences (each element is a full sequence)
      Y_object: list/array of corresponding label sequences
      sequence_length: number of past timesteps to use as input
      prediction_length: number of timesteps to predict
      
    Returns:
      Tuple of (X_past, Y_object_past, Y_object_future, sequence_length)
    """
    X_past = []
    Y_object_past = []
    Y_object_future = []

    n_past = sequence_length
    n_pred = prediction_length

    for combined_seq, y_seq in zip(X_combined, Y_object):
        combined_seq = np.array(combined_seq)
        y_seq = np.array(y_seq)

        if len(combined_seq) < n_past + n_pred or len(y_seq) < n_past + n_pred:
            print(f"Skipping short sequence: combined_seq length {len(combined_seq)}, y_seq length {len(y_seq)}")
            continue

        # Slide a window over the sequence
        for start in range(0, len(combined_seq) - n_past - n_pred + 1):
            end_past = start + n_past
            end_pred = end_past + n_pred

            try:
                X_past.append(combined_seq[start:end_past])
                Y_object_past.append(y_seq[start:end_past])
                Y_object_future.append(y_seq[end_past:end_pred])
            except IndexError as e:
                print(f"Slicing error at index {start}: {e}")
                continue

    return (
        np.array(X_past, dtype=object),
        np.array(Y_object_past, dtype=object),
        np.array(Y_object_future, dtype=object),
        sequence_length,
    )

# For CSV data we treat the entire dataset as one long sequence
X_combined = [X_scaled]
Y_object = [y]
sequence_length = 15
prediction_length = 1

X_past, Y_object_past, Y_object_future, seq_len = create_reference(X_combined, Y_object, sequence_length, prediction_length)
# Squeeze future labels (prediction_length==1) to shape (n_samples,)
print(Y_object_future.shape)
Y_future = np.squeeze(Y_object_future, axis=1)

# ------------------------------------------------------
# 3. Train-Test Split and Tensor Conversion
# ------------------------------------------------------
# Split the sequences into training and test sets
X_train, X_test, Y_past_train, Y_past_test, y_train, y_test = train_test_split(
    X_past, Y_object_past, Y_future, test_size=0.2, random_state=42
)

# Convert from arrays of objects to regular NumPy arrays (each sequence becomes a 2D array)
X_train_arr = np.array([np.array(seq) for seq in X_train])
X_test_arr = np.array([np.array(seq) for seq in X_test])
Y_past_train_arr = np.array([np.array(seq) for seq in Y_past_train])
Y_past_test_arr = np.array([np.array(seq) for seq in Y_past_test])

# Convert to TensorFlow tensors
X_train_tensor = tf.convert_to_tensor(X_train_arr, dtype=tf.float32)
Y_past_train_tensor = tf.convert_to_tensor(Y_past_train_arr, dtype=tf.int32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.int32)

X_test_tensor = tf.convert_to_tensor(X_test_arr, dtype=tf.float32)
Y_past_test_tensor = tf.convert_to_tensor(Y_past_test_arr, dtype=tf.int32)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.int32)

# Create TensorFlow datasets yielding ((X, Y_past), label)
train_dataset = (tf.data.Dataset.from_tensor_slices(((X_train_tensor, Y_past_train_tensor), y_train_tensor))
                 .shuffle(1000)
                 .batch(16)
                 .prefetch(tf.data.AUTOTUNE))
test_dataset = (tf.data.Dataset.from_tensor_slices(((X_test_tensor, Y_past_test_tensor), y_test_tensor))
                .batch(16)
                .prefetch(tf.data.AUTOTUNE))

# ------------------------------------------------------
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
                                   dropout=0.3, recurrent_dropout=0.)
        )
        self.attention = tf.keras.layers.Attention()
        self.state_h_dense = tf.keras.layers.Dense(self.hidden_state_size)
        self.state_c_dense = tf.keras.layers.Dense(self.hidden_state_size)
        self.query_dense = tf.keras.layers.Dense(self.hidden_state_size * 2)

        # Decoder and Output
        self.decoder = tf.keras.layers.LSTM(self.hidden_state_size, return_sequences=True, return_state=True,
                                            dropout=0.3, recurrent_dropout=0.)
        self.output_layer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def build(self, input_shape):
        """
        Overrides build() to ensure child layers are built properly.
        If input_shape is None (or its first element is None), we simply call the parent's build.
        """
        if input_shape is None or input_shape[0] is None:
            super(ActionPredictionModel, self).build(input_shape)
            return

        # Expecting input_shape as a tuple: (X_seq_shape, Y_past_shape)
        feature_shape = input_shape[0]  # e.g., (batch_size, sequence_length, feature_dim)
        self.norm_layer.build(feature_shape)
        self.feature_dense1.build(feature_shape)
        # Assume the output of feature_dense1 has last dim = 32.
        self.feature_dense2.build((feature_shape[0], feature_shape[1], 32))
        self.feature_dense3.build((feature_shape[0], feature_shape[1], 16))
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
        
        # Cast Y_past to float16 so that both x and Y_past_cast have the same dtype.
        Y_past_cast = tf.expand_dims(tf.cast(Y_past, tf.float16), axis=-1)
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
# ------------------------------------------------------
# 5. Training Utilities
# ------------------------------------------------------
def train_step(model, optimizer, loss_fn, inputs, labels):
    X_batch, Y_past_batch = inputs
    with tf.GradientTape() as tape:
        predictions = model((X_batch, Y_past_batch), training=True)
        # Squeeze predictions if they have an extra time dimension (shape becomes (batch, vocab_size))
        predictions = tf.squeeze(predictions, axis=1)
        # Labels are already shape (batch,), so use them directly.
        labels_onehot = tf.one_hot(labels, depth=model.parameters['vocab_size'])
        loss = loss_fn(labels_onehot, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

def evaluate_model(model, dataset):
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
    for (X_batch, Y_past_batch), labels in dataset:
        predictions = model((X_batch, Y_past_batch), training=False)
        predictions = tf.squeeze(predictions, axis=1)
        # Use labels directly without squeezing.
        labels_onehot = tf.one_hot(labels, depth=model.parameters['vocab_size'])
        accuracy_metric.update_state(labels_onehot, predictions)
    return accuracy_metric.result().numpy()

def train_model(model, train_dataset, test_dataset, epochs, optimizer, loss_fn):
    for epoch in range(epochs):
        epoch_loss = 0
        accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
        for (X_batch, Y_past_batch), labels in train_dataset:
            loss, predictions = train_step(model, optimizer, loss_fn, (X_batch, Y_past_batch), labels)
            epoch_loss += loss
            labels_onehot = tf.one_hot(labels, depth=model.parameters['vocab_size'])
            accuracy_metric.update_state(labels_onehot, predictions)
        val_accuracy = evaluate_model(model, test_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss.numpy():.4f}, "
              f"Train Accuracy: {accuracy_metric.result().numpy():.4f}, Validation Accuracy: {val_accuracy:.4f}")
# ------------------------------------------------------
# 6. Model Initialization and Training
# ------------------------------------------------------
# Set vocab_size based on the unique labels in the original dataset.
vocab_size = len(np.unique(Y_object[0]))
parameters = {'sequence_length': sequence_length, 'vocab_size': vocab_size, 'hidden_state_size': 20}
model = ActionPredictionModel(parameters)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Train the model
train_model(model, train_dataset, test_dataset, epochs=25, optimizer=optimizer, loss_fn=loss_fn)