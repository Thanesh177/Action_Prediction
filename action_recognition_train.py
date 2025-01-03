from __future__ import print_function
import sys
import os
import tensorflow as tf
import action_recognition_rnn as model
import action_dataset as dataset
import mirko_dataset as mirko_dataset

import numpy as np
from sklearn.utils import shuffle
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=3, threshold=sys.maxsize)

def clip_to_zero(x):
    idxs = np.absolute(x) < 1e-3
    x[idxs] = 0
    return x

initial_time = int(time.time())

log_id = int(time.time())
print("Log ID: ", log_id)

ACTION = 1
MIRKO = 2
DATASET = MIRKO

training_epochs = 50
batch_size = 100
INPUT_DROP = 0.3  # Reduced dropout to retain more features
RECUR_DROP = 0.05  # Reduced recurrent dropout for better gradient flow
OUTPUT_DROP = 0.0

if DATASET == ACTION:
    SEQLEN = 50
    FEATURE_SIZE = 60
    X_train, X_test, y_train, y_test = dataset.load()
    seq_features, seq_labels = dataset.load_sequence(1)

if DATASET == MIRKO:
    DOWNSAMPLING_STEP = 1
    WINDOW = 100
    FEATURE_SIZE = 142
    SEQLEN = int(WINDOW / DOWNSAMPLING_STEP)

dataset_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet'

params = {'sampling': 2, 'window': 100, 'window_step': 10}  # Define the dictionary
X_train, X_test, y_train, y_test = mirko_dataset.load(params, dataset_path)


print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)
print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)



# Updated RecognitionRNN class with checkpointing
class RecognitionRNN(tf.keras.Model):
    def __init__(self, batch_size, input_drop_prob, recurr_drop_prob, output_drop_prob, parameters):
        super(RecognitionRNN, self).__init__()
        self.batch_size = batch_size
        self.input_drop_prob = input_drop_prob
        self.recurr_drop_prob = recurr_drop_prob
        self.output_drop_prob = output_drop_prob
        self.seqlen = parameters["seqlen"]

        self.input_dropout = tf.keras.layers.Dropout(self.input_drop_prob)
        self.rnn = tf.keras.layers.LSTM(512, return_sequences=True, dropout=self.recurr_drop_prob)
        self.output_dense = tf.keras.layers.Dense(8, activation=None)

        # Checkpoint attributes
        self.checkpoint = tf.train.Checkpoint(model=self)
        self.checkpoint_manager = None

    def build(self, input_shape):
        # Initialize weights based on the input shape
        self.input_dense = tf.keras.layers.Dense(input_shape[-1], activation="relu")
        super(RecognitionRNN, self).build(input_shape)

    def call(self, inputs, training=False):
        x = self.input_dropout(inputs, training=training)
        x = self.rnn(x, training=training)
        return self.output_dense(x, training=training)

    def compute_loss(self, logits, labels):
        valid_mask = tf.cast(labels >= 0, tf.float32)
        masked_labels = tf.where(labels >= 0, labels, tf.zeros_like(labels))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masked_labels, logits=logits)
        loss = tf.reduce_sum(loss * valid_mask) / (tf.reduce_sum(valid_mask) + 1e-8)
        return loss

    def compute_accuracy(self, logits, labels):
        predictions = tf.argmax(logits, axis=-1)
        valid_mask = tf.cast(labels >= 0, tf.float32)
        accuracy = tf.reduce_sum(tf.cast(predictions == labels, tf.float32) * valid_mask) / (tf.reduce_sum(valid_mask) + 1e-8)
        return accuracy

    def initialize_checkpoint_manager(self, checkpoint_dir="checkpoints"):
        # Initialize a checkpoint manager for saving and restoring
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=checkpoint_dir, max_to_keep=5)

    def save_checkpoint(self):
        if self.checkpoint_manager is None:
            raise ValueError("Checkpoint manager is not initialized. Call 'initialize_checkpoint_manager' first.")
        save_path = self.checkpoint_manager.save()
        print(f"Checkpoint saved at: {save_path}")

# Initialize the model
params = {"seqlen": SEQLEN}
motionmodel = RecognitionRNN(
    batch_size=batch_size,
    input_drop_prob=INPUT_DROP,
    recurr_drop_prob=RECUR_DROP,
    output_drop_prob=OUTPUT_DROP,
    parameters=params,
)

# Initialize Checkpoint Manager
motionmodel.initialize_checkpoint_manager(checkpoint_dir="checkpoints")

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def call(self, query, value):
        scores = tf.matmul(query, value, transpose_b=True)
        distribution = tf.nn.softmax(scores, axis=-1)
        attention = tf.matmul(distribution, value)
        return attention

# Add Attention Mechanism to the Model
motionmodel.attention_layer = AttentionLayer()

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)  # Adjusted learning rate for stability

# Scheduler for dynamic learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-4,
    decay_steps=1000,
    decay_rate=0.95,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Training Loop
for epoch in range(training_epochs):
    X_train, y_train = shuffle(X_train, y_train)

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        with tf.GradientTape() as tape:
            logits = motionmodel(X_batch, training=True)

            # Add attention mechanism
            query = logits
            value = logits
            attention_output = motionmodel.attention_layer(query, value)

            # Use attention output for loss computation
            loss_value = motionmodel.compute_loss(attention_output, y_batch)

        gradients = tape.gradient(loss_value, motionmodel.trainable_variables)
        optimizer.apply_gradients(zip(gradients, motionmodel.trainable_variables))

    # Evaluate after epoch
    logits_train = motionmodel(X_train, training=False)
    train_loss = motionmodel.compute_loss(logits_train, y_train)
    train_accuracy = motionmodel.compute_accuracy(logits_train, y_train)

    logits_test = motionmodel(X_test, training=False)
    test_loss = motionmodel.compute_loss(logits_test, y_test)
    test_accuracy = motionmodel.compute_accuracy(logits_test, y_test)

    print(f"Epoch {epoch + 1}/{training_epochs}")
    print(f"Train Loss: {train_loss.numpy():.4f}, Train Accuracy: {train_accuracy.numpy():.4f}")
    print(f"Test Loss: {test_loss.numpy():.4f}, Test Accuracy: {test_accuracy.numpy():.4f}")

    # Save checkpoint periodically
    if (epoch + 1) % 10 == 0:
        motionmodel.save_checkpoint()

# Final evaluation
logits_test = motionmodel(X_test, training=False)
y_hat = tf.argmax(logits_test, axis=-1).numpy()

print("Predicted y_hat:", y_hat[:10])
print("True labels:", y_test[:10])

final_time = time.time()
print(f"Training completed in {(final_time - initial_time) / 60:.2f} minutes")
