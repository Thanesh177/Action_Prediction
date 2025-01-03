import numpy as np
import scipy.io
import random

def onehot(tensor, num_labels):
    return np.eye(num_labels)[tensor]

def reconstructStructure(joints, body, gaze, merged_cuts):
    num_samples = joints.shape[2]

    X_object_ = []
    X_body_ = []
    Y_object_ = []
    X_gaze_ = []

    for idx in range(num_samples):
        obj_features = joints[:, :, idx].T
        body_features = body[:, :, idx].T
        gaze_features = gaze[:, :, idx].T

        # Extract the first column as the label
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

    print("Final Y_object_ content after validation:", Y_object_[:2])
    return (
        np.array(X_object_, dtype=object),
        np.array(X_body_, dtype=object),
        np.array(X_gaze_, dtype=object),
        np.array(Y_object_, dtype=object),
    )

def loadMatDataset(path):
    # Load the .mat dataset and return its contents
    return scipy.io.loadmat(path)

def load(dataset_path, sequence_length, prediction_length):
    # Load the .mat file
    mat_data = loadMatDataset(dataset_path)

    gaze = mat_data['gaze']
    merged_cuts = mat_data['merged_cuts']
    joints = mat_data['joints']
    body = mat_data['body']
    print(mat_data.keys())  # Check keys in the dataset
    print("merged_cuts shape:", mat_data['merged_cuts'].shape)
    print("gaze:", gaze[:1])  # Print a few rows to inspect
    print("joints sample:", joints[:1])  # Print a few rows to inspect
    print("body sample:", body[:1])  # Print a few rows to inspect
    print("gaze sample:", gaze.shape)  # Print a few rows to inspect
    print("joints sample:", joints.shape)  # Print a few rows to inspect
    print("body sample:", body.shape)  # Print a few rows to inspect



    # Reconstruct structured data
    X_object_, X_body_, X_gaze_, Y_object_ = reconstructStructure(joints, body, gaze, merged_cuts)

    # Create sequences and references
    X_object_, X_body_, X_gaze_, Y_object_past_, Y_object_, sequence_lengths = create_reference(
        X_object_, X_body_, X_gaze_, Y_object_, sequence_length, prediction_length
    )

    return X_object_, X_body_, X_gaze_, Y_object_past_, Y_object_, sequence_lengths

def create_reference(X_object, X_body, X_gaze, Y_object, sequence_length, prediction_length):
    X_object_ = []
    X_body_ = []
    X_gaze_ = []
    Y_object_ = []
    Y_object_past_ = []

    n_past = sequence_length
    n_pred = prediction_length

    for obj_seq, body_seq, gaze_seq, y_seq in zip(X_object, X_body, X_gaze, Y_object):
        obj_seq = np.array(obj_seq)
        body_seq = np.array(body_seq)
        gaze_seq = np.array(gaze_seq)
        y_seq = np.array(y_seq)  # Ensure y_seq is an array

        if len(obj_seq) < n_past + n_pred or len(y_seq) < n_past + n_pred:
            print(f"Skipping short sequence: obj_seq length {len(obj_seq)}, y_seq length {len(y_seq)}")
            continue

        for start in range(0, len(obj_seq) - n_past - n_pred + 1):
            end_past = start + n_past
            end_pred = end_past + n_pred

            try:
                X_object_.append(obj_seq[start:end_past])
                X_body_.append(body_seq[start:end_past])
                X_gaze_.append(gaze_seq[start:end_past])
                Y_object_past_.append(y_seq[start:end_past])
                Y_object_.append(y_seq[end_past:end_pred])
            except IndexError as e:
                print(f"Slicing error at index {start}: {e}")
                continue

    print("Final Y_object_ after create_reference:", Y_object_[:10])
    print("Y_object_ shape after create_reference:", len(Y_object_))

    return (
        np.array(X_object_, dtype=object),
        np.array(X_body_, dtype=object),
        np.array(X_gaze_, dtype=object),
        np.array(Y_object_past_, dtype=object),
        np.array(Y_object_, dtype=object),
        sequence_length,
    )

def sampleSubSequences(length, num_samples=1, min_len=1, max_len=10):
    max_len = min(max_len, length)
    min_len = min(min_len, max_len)
    sequence = []
    for _ in range(num_samples):
        l = random.randint(min_len, max_len)
        start_idx = random.randint(0, length - l)
        end_idx = start_idx + l
        sequence.append((start_idx, end_idx))

    return sequence

if __name__ == "__main__":
    dataset_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/merged_labeled_actions.mat'
    sequence_length = 10
    prediction_length = 5

    X_object_, X_body_, X_gaze_, Y_object_past_, Y_object_, sequence_lengths = load(
        dataset_path, sequence_length, prediction_length
    )

