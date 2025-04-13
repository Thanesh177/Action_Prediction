import cv2
import numpy as np
import copy
from network import test_dataset, model, GAZE_MAX, GAZE_MIN

last_predicted = None  # Global variable to store last predicted gaze.

def draw_gaze(image, gaze):
    """
    Draws a circle at the given gaze coordinates on a copy of the image.
    """
    canvas = copy.deepcopy(image)
    center = (int(gaze[0]), int(gaze[1]))
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
    Returns denormalized gaze coordinates (shape: (2,)) for the prediction.
    """
    (X_combined_batch, Y_past_batch), _ = test_sample
    gaze_pred = model((X_combined_batch, Y_past_batch), training=False)
    # gaze_pred now has shape (batch, 2)
    predicted_gaze_norm = gaze_pred[0].numpy()  # (2,)
    print("Predicted normalized gaze:", predicted_gaze_norm)
    predicted_gaze_norm = np.clip(predicted_gaze_norm, 0.0, 1.0)
    predicted_gaze = predicted_gaze_norm * (GAZE_MAX - GAZE_MIN) + GAZE_MIN
    return predicted_gaze

def show_videos():
    """
    Opens the world video and overlays both the ground-truth gaze and the predicted gaze.
    Draws a light trail (a bright line) from the previous predicted position to the current predicted position.
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

        # For demonstration, use the first label repeatedly.
        current_label = labels[0]
        gt_gaze = current_label[-2:]
        
        try:
            test_sample = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_dataset)
            test_sample = next(test_iterator)
        
        # Predict gaze.
        predicted_gaze = predict_gaze_from_test_sample(model, test_sample)
        
        # Draw ground-truth and predicted gaze.
        frame_gt = draw_gaze(frame, gt_gaze)
        frame_pred = draw_gaze(frame, predicted_gaze)
        
        # Draw a light trail (bright line) from last predicted to current predicted gaze.
        if last_predicted is not None:
            pt1 = (int(last_predicted[0]), int(last_predicted[1]))
            pt2 = (int(predicted_gaze[0]), int(predicted_gaze[1]))
            cv2.line(frame_pred, pt1, pt2, (0, 255, 255), thickness=2)
        
        # Update last predicted gaze.
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