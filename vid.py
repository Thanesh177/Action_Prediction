import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io
from dataset import load  # Assumes load() returns: X_combined_, Y_object_past_, Y_action_, Y_gaze_
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import copy
from network import test_dataset
from dataset import load  # Assumes load() returns: X_combined_, Y_object_past_, Y_action_, Y_gaze_
from network import model  # Assumes this is your model class


dataset_path = '/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/merged_labeled_actions.mat'
sequence_length = 4
prediction_length = 1

# --- (Optional) Compute clustering on joint data ---
mat_data = scipy.io.loadmat(dataset_path)
joints_data = mat_data['joints']
joint_variances = np.var(joints_data, axis=(1, 2))
mean_positions = np.mean(joints_data, axis=2)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
joint_clusters = kmeans.fit_predict(mean_positions)
target_cluster = 4

X_combined_, Y_object_past_, Y_action_, Y_gaze_ = load(
    dataset_path, sequence_length, prediction_length, joint_clusters, target_cluster=target_cluster
)
# --- Normalize gaze targets ---
GAZE_MIN = np.min(Y_gaze_)
GAZE_MAX = np.max(Y_gaze_)
Y_gaze_norm = (Y_gaze_ - GAZE_MIN) / (GAZE_MAX - GAZE_MIN)

print(f"GAZE_MIN: {GAZE_MIN}, GAZE_MAX: {GAZE_MAX}")

# Assume these are defined globally from your training/preprocessing code:
# X_train_tensor, GAZE_MIN, GAZE_MAX, and model (your trained model)
def draw_gaze(image, gaze):
    """
    Draws a circle at the given gaze coordinates on a copy of the image.
    """
    canvas = copy.deepcopy(image)
    center = (int(gaze[0]), int(gaze[1]))
    # Only draw the marker if valid.
    if -1 not in center:
        cv2.circle(canvas, center, 10, (0, 255, 0), -1)
    return canvas

def draw_class(image, label):
    """
    Overlays the action class text on the image.
    """
    actions = {
        1: "Place Left",
        2: "Give Left",
        3: "Place Middle",
        4: "Give Middle",
        5: "Place Right",
        6: "Give Right",
        7: "Pick Left",
        8: "Receive Left",
        9: "Pick Middle",
        10: "Receive Middle",
        11: "Pick Right",
        12: "Receive Right"
    }
    canvas = copy.deepcopy(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (image.shape[1]//2 - 150, image.shape[0]//2)
    fontScale = 1.3
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(canvas, "{} ({})".format(actions[label], label),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    return canvas
last_predicted = None  # Global variable to store last predicted gaze.

def draw_gaze(image, gaze):
    """
    Draws a circle at the given gaze coordinates on a copy of the image.
    """
    canvas = copy.deepcopy(image)
    # Convert gaze coordinates to Python scalars
    center = (int(gaze[0].item() if hasattr(gaze[0], 'item') else gaze[0]),
              int(gaze[1].item() if hasattr(gaze[1], 'item') else gaze[1]))
    if -1 not in center:
        cv2.circle(canvas, center, 10, (0, 255, 0), -1)
    return canvas

def draw_class(image, label):
    """
    Overlays the action class text on the image.
    """
    actions = {
        1: "Place Left",
        2: "Give Left",
        3: "Place Middle",
        4: "Give Middle",
        5: "Place Right",
        6: "Give Right",
        7: "Pick Left",
        8: "Receive Left",
        9: "Pick Middle",
        10: "Receive Middle",
        11: "Pick Right",
        12: "Receive Right"
    }
    canvas = copy.deepcopy(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (image.shape[1] // 2 - 150, image.shape[0] // 2)
    fontScale = 1.3
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(canvas, "{} ({})".format(actions[label], label),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    return canvas

def predict_gaze_from_test_sample(model, test_sample):
    """
    Uses a sample from the test dataset to predict gaze.
    Returns denormalized gaze coordinates (shape: (2,)) for the third prediction step.
    """
    (X_combined_batch, Y_past_batch), _ = test_sample
    gaze_pred, _, _ = model((X_combined_batch, Y_past_batch), training=False)
    # gaze_pred now has shape (batch, 2); take the first sample and squeeze it to ensure it's 1D.
    predicted_gaze_norm = np.squeeze(gaze_pred[0].numpy())  # should have shape (2,)
    print("Predicted normalized gaze:", predicted_gaze_norm)
    predicted_gaze_norm = np.clip(predicted_gaze_norm, 0.0, 1.0)
    predicted_gaze = predicted_gaze_norm * (GAZE_MAX - GAZE_MIN) + GAZE_MIN
    return predicted_gaze
def show_videos():
    """
    Opens the world video and overlays both the ground-truth gaze and the predicted gaze.
    Draws a light trail (a bright line) from the previous predicted position to the current predicted position,
    and displays the predicted gaze coordinates on the video.
    """
    global last_predicted
    cap_world = cv2.VideoCapture("/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/NewLabeledVideo/world.mp4")
    labels = np.load("/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/NewLabeledVideo/action_labels.npy")
    
    # Create an iterator over the test dataset.
    test_iterator = iter(test_dataset)
    
    while True:
        ret, frame = cap_world.read()
        if not ret:
            break

        # For demonstration, we use the first label repeatedly (in practice, synchronize labels with frames).
        current_label = labels[0]
        gt_gaze = current_label[-2:]
        
        try:
            test_sample = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_dataset)
            test_sample = next(test_iterator)
        
        # Predict gaze.
        predicted_gaze = predict_gaze_from_test_sample(model, test_sample)
        
        # Draw markers.
        frame_gt = draw_gaze(frame, gt_gaze)
        frame_pred = draw_gaze(frame, predicted_gaze)
        
        # Draw a light trail (bright line) from the previous predicted to the current predicted gaze.
        if last_predicted is not None:
            pt1 = (int(last_predicted[0]), int(last_predicted[1]))
            pt2 = (int(predicted_gaze[0]), int(predicted_gaze[1]))
            cv2.line(frame_pred, pt1, pt2, (0, 255, 255), thickness=2)
        
        # Display the predicted gaze value as text on the predicted frame.
        pred_text = "Pred: ({:.1f}, {:.1f})".format(predicted_gaze[0], predicted_gaze[1])
        cv2.putText(frame_pred, pred_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Update the last predicted gaze.
        last_predicted = predicted_gaze
        
        # Concatenate ground-truth and predicted frames side-by-side.
        frame_concat = np.concatenate((frame_gt, frame_pred), axis=1)
        # Overlay the action class using current_label[1].
        frame_concat = draw_class(frame_concat, current_label[1])
        cv2.imshow("Action Dataset", frame_concat)
        if cv2.waitKey(25) == 27:  # Press ESC to exit.
            break

    cap_world.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    show_videos()