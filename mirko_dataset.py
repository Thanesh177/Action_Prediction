import numpy as np
import os
import scipy.io
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Global Parameters
DOWNSAMPLING_STEP = 1
WINDOW = 600
WINDOW_STEP = 1
N_COMPONENTS = None  # Number of PCA components for feature reduction

# Dataset Path
dataset_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet'

def readMat(path):
    """Reads the dataset from the .mat file."""
    features_path = os.path.join(path, 'merged_labeled_actions.mat')

    # Load the .mat file
    mat = scipy.io.loadmat(features_path)

    # Extract features
    joint_features = mat['joints'].transpose(2, 1, 0)  # Shape: (samples, timesteps, features)
    gaze_features = mat['gaze'].transpose(2, 1, 0)  # Shape: (samples, timesteps, features)
    print(gaze_features[1])
    # Combine features along the last axis

    # Flatten and fit a model
    gaze_flat = gaze_features.reshape(-1, 3)
    joint_flat = joint_features.reshape(-1, 39)
    lr = LinearRegression().fit(gaze_flat, joint_flat)
    gaze_transformed = lr.predict(gaze_flat).reshape(joint_features.shape)

    combined_features = (joint_features * gaze_transformed) / 2
    features = np.concatenate((combined_features, joint_features, gaze_transformed), axis=2)

    # Flatten features into 2D array
    features = features.reshape(features.shape[0] * features.shape[1], features.shape[2])

    # Extract labels
    labels = mat['merged_cuts'][:, 1]  # Assuming column 1 represents class labels

    # Reshape labels to 2D if they are 1D
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    # Normalize and apply PCA
    features = normalize_features(features)
    features = apply_pca(features, n_components=N_COMPONENTS)

    # Reshape features back to 3D after PCA
    if N_COMPONENTS:
        features = features.reshape(-1, mat['joints'].shape[1], N_COMPONENTS)
    else:
        features = features.reshape(mat['joints'].shape[2], mat['joints'].shape[1], -1)

    return features, labels


def normalize_features(features):
    """Normalizes the features while preserving zero values."""
    mask = features == 0
    scaler = preprocessing.StandardScaler()
    non_zero_features = features[~mask].reshape(-1, 1)  # Ensure 2D for StandardScaler
    scaler.fit(non_zero_features)
    features[~mask] = scaler.transform(non_zero_features).flatten()  # Flatten back to original shape
    return features


def apply_pca(features, n_components):
    """Reduces the dimensionality of features using PCA."""
    if n_components is None or n_components > min(features.shape):
        return features
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features)
    return features_reduced


def downsample(sequence, step):
    """Downsamples the sequence by the specified step."""
    idxs = list(range(0, sequence.shape[0], step))
    return sequence[idxs]


def cutUp(features, labels, window_size, step, sampling_step):
    """Cuts up the features and labels into fixed-size windows."""
    window_samples = int(window_size / sampling_step)
    feature_segments = []
    label_segments = []

    for feature_vectors, label in zip(features, labels):
        length = np.count_nonzero(feature_vectors[:, 0])  # Count non-zero entries in the first column
        for i in range(0, length - window_size - 1, step):
            segment = downsample(feature_vectors[i:i + window_size, :], sampling_step)
            feature_segments.append(segment)
            label_segments.append(np.resize(label, (window_samples)))

    return np.array(feature_segments, dtype=np.float32), np.array(label_segments, dtype=np.int32)


def load(params, dataset_path):
    global DOWNSAMPLING_STEP, WINDOW, WINDOW_STEP, N_COMPONENTS

    if params and isinstance(params, dict):  # Ensure `params` is a dictionary
        DOWNSAMPLING_STEP = params.get('sampling', 1)
        WINDOW = params.get('window', 600)
        WINDOW_STEP = params.get('window_step', 1)
        N_COMPONENTS = params.get('n_components', None)

    features, labels = readMat(dataset_path)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=33)

    # Cut into windows
    X_train, y_train = cutUp(X_train, y_train, WINDOW, WINDOW_STEP, DOWNSAMPLING_STEP)
    X_test, y_test = cutUp(X_test, y_test, WINDOW, WINDOW_STEP, DOWNSAMPLING_STEP)

    # Shuffle data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    return X_train, X_test, y_train, y_test


def load_sequence(index, dataset_path):
    """Loads a sequence for evaluation."""
    features, labels = readMat(dataset_path)

    # Downsample sequences
    sequence_features = downsample(features, DOWNSAMPLING_STEP)

    # Expand labels to match sequence length
    labels = np.expand_dims(labels, axis=1)
    sequence_labels = np.repeat(labels, sequence_features.shape[1], axis=1)

    return sequence_features, sequence_labels
