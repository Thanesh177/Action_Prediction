import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
import ruptures as rpt

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("JIT compilation not supported on this machine.")

tf.keras.mixed_precision.set_global_policy('mixed_float16')


file_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/prediction/merge.csv'
data = pd.read_csv(file_path, skiprows=1, header=None)
data.columns = ['Head_Vel', 'HeadVector_X', 'HeadVector_Y', 'HeadVector_Z', 'Label']

X = data[['Head_Vel', 'HeadVector_X', 'HeadVector_Y', 'HeadVector_Z']].values
y = data['Label'].astype(int).values  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def downsample_cpd(X, y, model="l2", penalty=2):
    algo = rpt.Pelt(model=model).fit(X)
    change_points = algo.predict(pen=penalty)
    indices = [0] 
    for cp in change_points[:-1]:
        indices.append(cp)
    indices = sorted(set(indices))
    return X[indices], y[indices]

X_downsampled, y_downsampled = downsample_cpd(X_scaled, y, model='l2', penalty=2)
print(f"Original number of frames: {len(X_scaled)}, Downsampled: {len(X_downsampled)}")


def create_sequences(X, y, sequence_length):
    X_seq, Y_past, y_target = [], [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i: i+sequence_length])
        Y_past.append(y[i: i+sequence_length])
        y_target.append(y[i+sequence_length])
    return np.array(X_seq), np.array(Y_past), np.array(y_target)

sequence_length = 10
X_seq, Y_past, y_target = create_sequences(X_downsampled, y_downsampled, sequence_length)

X_train, X_test, Y_past_train, Y_past_test, y_train, y_test = train_test_split(
    X_seq, Y_past, y_target, test_size=0.2, random_state=42, shuffle=True
)

batch_size = 10
train_dataset = tf.data.Dataset.from_tensor_slices(((X_train, Y_past_train), y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices(((X_test, Y_past_test), y_test))
test_dataset = test_dataset.batch(batch_size)


class ActionPredictionModel(tf.keras.Model):
    def __init__(self, parameters):
        super(ActionPredictionModel, self).__init__()
        self.parameters = parameters
        self.hidden_state_size = parameters['hidden_state_size']
        self.vocab_size = parameters['vocab_size']

        self.norm_layer = tf.keras.layers.LayerNormalization()
        self.feature_dense1 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-4))
        self.feature_dense2 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(1e-4))
        self.feature_dense3 = tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2(1e-4))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.batch_norm = tf.keras.layers.BatchNormalization()

        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_state_size, return_sequences=True, return_state=True,
                                 dropout=0.3, recurrent_dropout=0.3)
        )
        self.attention = tf.keras.layers.Attention()
        self.state_h_dense = tf.keras.layers.Dense(self.hidden_state_size)
        self.state_c_dense = tf.keras.layers.Dense(self.hidden_state_size)
        self.query_dense = tf.keras.layers.Dense(self.hidden_state_size * 2)
        self.decoder = tf.keras.layers.LSTM(self.hidden_state_size, return_sequences=True, return_state=True,
                                            dropout=0.3, recurrent_dropout=0.3)
        self.output_layer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, training=False):
        X_seq, Y_past = inputs  

        X_seq = self.norm_layer(X_seq)
        combined_embedding = self.feature_dense1(X_seq)
        combined_embedding = self.batch_norm(combined_embedding, training=training)
        combined_embedding = self.dropout(combined_embedding, training=training)
        combined_embedding = self.feature_dense2(combined_embedding)
        combined_embedding = self.dropout(combined_embedding, training=training)
        combined_embedding = self.feature_dense3(combined_embedding)

        Y_past = tf.expand_dims(tf.cast(Y_past, tf.float16), axis=-1)

        input_sequence = tf.concat([combined_embedding, Y_past], axis=2)

        encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder(input_sequence)
        state_h = tf.concat([forward_h, backward_h], axis=-1)
        state_c = tf.concat([forward_c, backward_c], axis=-1)
        state_h = self.state_h_dense(state_h)
        state_c = self.state_c_dense(state_c)

        query = self.query_dense(state_h)
        query = tf.expand_dims(query, axis=1)
        attention_output = self.attention([query, encoder_outputs])

        decoder_outputs, _, _ = self.decoder(attention_output, initial_state=[state_h, state_c])
        predictions = self.output_layer(decoder_outputs)
        return predictions


def train_step(model, optimizer, loss_fn, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        labels_onehot = tf.one_hot(labels, depth=model.parameters['vocab_size'])
        predictions = tf.squeeze(predictions, axis=1)
        loss = loss_fn(labels_onehot, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

def evaluate_model(model, dataset):
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
    for (X_batch, Y_past_batch), labels in dataset:
        predictions = model((X_batch, Y_past_batch), training=False)
        labels_onehot = tf.one_hot(labels, depth=model.parameters['vocab_size'])
        accuracy_metric.update_state(labels_onehot, predictions)
    return accuracy_metric.result().numpy()

def train_model(model, train_dataset, test_dataset, epochs, optimizer, loss_fn):
    for epoch in range(epochs):
        train_loss_metric = tf.keras.metrics.Mean()
        train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
        
        for (X_batch, Y_past_batch), labels in train_dataset:
            loss, predictions = train_step(model, optimizer, loss_fn, (X_batch, Y_past_batch), labels)
            train_loss_metric.update_state(loss)
            labels_onehot = tf.one_hot(labels, depth=model.parameters['vocab_size'])
            train_accuracy_metric.update_state(labels_onehot, predictions)
        
        val_loss_metric = tf.keras.metrics.Mean()
        val_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
        for (X_batch, Y_past_batch), labels in test_dataset:
            predictions = model((X_batch, Y_past_batch), training=False)
            labels_onehot = tf.one_hot(labels, depth=model.parameters['vocab_size'])
            predictions_squeezed = tf.squeeze(predictions, axis=1)
            loss = loss_fn(labels_onehot, predictions_squeezed)
            val_loss_metric.update_state(loss)
            val_accuracy_metric.update_state(labels_onehot, predictions_squeezed)
        
        train_loss = train_loss_metric.result().numpy()
        train_accuracy = train_accuracy_metric.result().numpy()
        val_loss = val_loss_metric.result().numpy()
        val_accuracy = val_accuracy_metric.result().numpy()
        
        print(f"Epoch {epoch+1}/{epochs} -- Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        



vocab_size = len(np.unique(y_downsampled))
parameters = {'sequence_length': sequence_length, 'vocab_size': vocab_size, 'hidden_state_size': 100}
model = ActionPredictionModel(parameters)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

train_model(model, train_dataset, test_dataset, epochs=100, optimizer=optimizer, loss_fn=loss_fn)
