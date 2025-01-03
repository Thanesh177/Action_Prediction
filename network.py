import tensorflow as tf
from dataset import load  # Adjust the import path based on your project structure

# Dataset parameters
dataset_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/merged_labeled_actions.mat'
sequence_length = 10
prediction_length = 5

# Load the dataset
X_object_, X_body_, X_gaze_, Y_object_past_, Y_object_, sequence_lengths = load(
    dataset_path, sequence_length, prediction_length
)

# Convert data to TensorFlow tensors
X_object_tensor = tf.ragged.constant(X_object_)
X_body_tensor = tf.ragged.constant(X_body_)
X_gaze_tensor = tf.ragged.constant(X_gaze_)
Y_object_past_tensor = tf.ragged.constant(Y_object_past_)
Y_object_tensor = tf.ragged.constant(Y_object_)

print("Y_object_ sample:", Y_object_[:1])
print("Y_object_ shape:", len(Y_object_))

# Define dataset
dataset = tf.data.Dataset.from_tensor_slices((
    X_object_tensor, X_body_tensor, X_gaze_tensor, Y_object_past_tensor, Y_object_tensor
)).batch(32)


# Model definition
class ActionPredictionModel(tf.keras.Model):
    def __init__(self, parameters):
        super(ActionPredictionModel, self).__init__()
        self.parameters = parameters
        self.hidden_state_size = parameters.get('hidden_state_size', 128)

        # Scene Representation Layers
        self.object_dense1 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.object_dense2 = tf.keras.layers.Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.object_dense3 = tf.keras.layers.Dense(10, activation='relu')
        self.gaze_dense = tf.keras.layers.Dense(10, activation='relu')

        # Context Encoder
        self.encoder = tf.keras.layers.LSTM(
            self.hidden_state_size, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.3
        )

        # Decoder
        self.decoder = tf.keras.layers.LSTM(
            self.hidden_state_size, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.3
        )

        # Output Layer
        self.output_layer = tf.keras.layers.Dense(
            self.parameters['vocab_size'], activation='softmax'
        )

    def call(self, inputs, training=False):
        X_object, X_body, X_gaze, Y_past = inputs

        # Process object features
        obj_embedding = self.object_dense1(X_object)
        obj_embedding = tf.keras.layers.Dropout(0.3)(obj_embedding, training=training)
        obj_embedding = self.object_dense2(obj_embedding)
        obj_embedding = self.object_dense3(obj_embedding)
        scene_representation = tf.reduce_sum(obj_embedding, axis=2)

        # Process gaze features
        gaze_representation = self.gaze_dense(X_gaze)
        gaze_representation = tf.reduce_sum(gaze_representation, axis=2)
        gaze_representation = tf.expand_dims(gaze_representation, axis=2)

        # Combine features
        scene_representation = tf.expand_dims(scene_representation, axis=2)
        input_sequence = tf.concat([
            scene_representation,
            X_body,
            gaze_representation,
            tf.expand_dims(tf.cast(Y_past, tf.float32), axis=-1)
        ], axis=2)

        # Encode context
        context_outputs, context_state_h, context_state_c = self.encoder(input_sequence)

        # Decode predictions
        decoder_outputs, _, _ = self.decoder(context_outputs, initial_state=[context_state_h, context_state_c])

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
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()

    for X_object_batch, X_body_batch, X_gaze_batch, Y_past_batch, Y_batch in dataset:
        inputs = (
            X_object_batch.to_tensor(),
            X_body_batch.to_tensor(),
            X_gaze_batch.to_tensor(),
            Y_past_batch.to_tensor(),
        )
        labels = tf.one_hot(Y_batch.to_tensor(), depth=model.parameters['vocab_size'])
        predictions = model(inputs, training=False)
        sliced_predictions = predictions[:, -labels.shape[1]:, :]  # Slice predictions for the last prediction_length

        # Update metrics
        accuracy_metric.update_state(labels, sliced_predictions)
        precision_metric.update_state(labels, sliced_predictions)
        recall_metric.update_state(labels, sliced_predictions)

    return accuracy_metric.result().numpy(), precision_metric.result().numpy(), recall_metric.result().numpy()



def train_model(model, dataset, epochs, optimizer, loss_fn):
    for epoch in range(epochs):
        epoch_loss = 0
        accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
        precision_metric = tf.keras.metrics.Precision()
        recall_metric = tf.keras.metrics.Recall()

        for X_object_batch, X_body_batch, X_gaze_batch, Y_past_batch, Y_batch in dataset:
            inputs = (
                X_object_batch.to_tensor(),
                X_body_batch.to_tensor(),
                X_gaze_batch.to_tensor(),
                Y_past_batch.to_tensor(),
            )
            labels = Y_batch.to_tensor()

            # Perform a training step
            loss, sliced_predictions = train_step(model, optimizer, loss_fn, inputs, labels)
            epoch_loss += loss

            # Update metrics
            accuracy_metric.update_state(tf.one_hot(labels, depth=model.parameters['vocab_size']), sliced_predictions)
            precision_metric.update_state(tf.one_hot(labels, depth=model.parameters['vocab_size']), sliced_predictions)
            recall_metric.update_state(tf.one_hot(labels, depth=model.parameters['vocab_size']), sliced_predictions)

        # Compute epoch metrics
        epoch_accuracy = accuracy_metric.result().numpy()
        epoch_precision = precision_metric.result().numpy()
        epoch_recall = recall_metric.result().numpy()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss.numpy()}, Accuracy: {epoch_accuracy}, Precision: {epoch_precision}, Recall: {epoch_recall}")


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
train_model(model, dataset, epochs, optimizer, loss_fn)

# Evaluate model accuracy, precision, and recall
accuracy, precision, recall = evaluate_model(model, dataset)
print(f"Final Training Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")