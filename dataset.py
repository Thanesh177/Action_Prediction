import numpy as np
import scipy.io
from sklearn.cluster import KMeans
import random

def onehot(tensor, num_labels):
    return np.eye(num_labels)[tensor]

def reconstructStructure(joints, body, gaze, merged_cuts):
    num_samples = joints.shape[2]

    X_object_ = []
    X_body_ = []
    X_gaze_ = []
    Y_object_ = []

    for idx in range(num_samples):
        obj_features = joints[:, :, idx].T
        body_features = body[:, :, idx].T
        gaze_features = gaze[:, :, idx].T

        # Extract the first column as the label from merged_cuts
        if idx < merged_cuts.shape[0]:
            label = merged_cuts[idx, 0]  # Assuming the first column contains labels
        else:
            label = 0  # Default label

        if label is None or np.isnan(label):  # Validate label
            print(f"Skipping invalid label at idx {idx}.")
            continue

        # Create a label sequence matching the length of obj_features
        label_sequence = np.array([int(label)] * obj_features.shape[0])

        X_object_.append(obj_features)
        X_body_.append(body_features)
        X_gaze_.append(gaze_features)
        Y_object_.append(label_sequence)

    return (
        np.array(X_object_, dtype=object),
        np.array(X_body_, dtype=object),
        np.array(X_gaze_, dtype=object),
        np.array(Y_object_, dtype=object),
    )

def loadMatDataset(path):
    # Load the .mat dataset and return its contents
    return scipy.io.loadmat(path)

def create_reference(X_combined, Y_object, sequence_length, prediction_length):
    X_combined_ = []
    Y_object_past_ = []
    Y_object_ = []

    n_past = sequence_length
    n_pred = prediction_length

    for combined_seq, y_seq in zip(X_combined, Y_object):
        combined_seq = np.array(combined_seq)
        y_seq = np.array(y_seq)  # Ensure y_seq is an array

        if len(combined_seq) < n_past + n_pred or len(y_seq) < n_past + n_pred:
            print(f"Skipping short sequence: combined_seq length {len(combined_seq)}, y_seq length {len(y_seq)}")
            continue

        for start in range(0, len(combined_seq) - n_past - n_pred + 1):
            end_past = start + n_past
            end_pred = end_past + n_pred

            try:
                X_combined_.append(combined_seq[start:end_past])
                Y_object_past_.append(y_seq[start:end_past])
                Y_object_.append(y_seq[end_past:end_pred])
            except IndexError as e:
                print(f"Slicing error at index {start}: {e}")
                continue

    return (
        np.array(X_combined_, dtype=object),
        np.array(Y_object_past_, dtype=object),
        np.array(Y_object_, dtype=object),
        sequence_length,
    )

def create_reference_gaze(X_gaze, sequence_length, prediction_length):
    """
    Create gaze target sequences from the list of gaze arrays.
    Each element in X_gaze is assumed to be a 2D array of shape (T, 2).
    We create sequences with a past window of length `sequence_length` and a prediction window of length `prediction_length`.
    Here we return only the target gaze sequence for each window.
    """
    Y_gaze_seq = []
    n_past = sequence_length
    n_pred = prediction_length

    for gaze_seq in X_gaze:
        gaze_seq = np.array(gaze_seq)
        if len(gaze_seq) < n_past + n_pred:
            print(f"Skipping short gaze sequence: length {len(gaze_seq)}")
            continue
        for start in range(0, len(gaze_seq) - n_past - n_pred + 1):
            # We take the target as the gaze data in the prediction window.
            Y_gaze_seq.append(gaze_seq[start + n_past : start + n_past + n_pred])
    return np.array(Y_gaze_seq, dtype=object)

def load(dataset_path, sequence_length, prediction_length, joint_clusters, target_cluster=4):
    # Load the .mat file
    mat_data = loadMatDataset(dataset_path)

    gaze = mat_data['gaze']
    merged_cuts = mat_data['merged_cuts']
    joints = mat_data['joints']
    body = mat_data['body']

    # Reconstruct structured data
    X_object_, X_body_, X_gaze_, Y_object_ = reconstructStructure(joints, body, gaze, merged_cuts)

    # Extract features for joints in the target cluster
    cluster_indices = np.where(np.atleast_1d(joint_clusters) == target_cluster)[0]
    X_object_ = [
        obj_seq[:, cluster_indices] for obj_seq in X_object_
    ]  # Extract hand movement features

    # Combine hand movement, body, and gaze features into one array for each sample.
    combined_features = [
        np.concatenate((obj_seq, body_seq, gaze_seq), axis=1)
        for obj_seq, body_seq, gaze_seq in zip(X_object_, X_body_, X_gaze_)
    ]

    # Create sequences for action labels.
    combined_features, Y_object_past_, Y_object_, _ = create_reference(
        combined_features, Y_object_, sequence_length, prediction_length
    )

    # For gaze, extract only the first two channels (2D coordinates) from each gaze sample.
    X_gaze_mod = [gaze_seq[:, :2] for gaze_seq in X_gaze_]
    # Create gaze target sequences.
    Y_gaze_ = create_reference_gaze(X_gaze_mod, sequence_length, prediction_length)

    return combined_features, Y_object_past_, Y_object_, Y_gaze_