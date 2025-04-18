import cv2
import numpy as np
import copy
from network import test_dataset, model, GAZE_MAX, GAZE_MIN

last_predicted = None 

def draw_gaze(image, gaze):

    canvas = copy.deepcopy(image)
    center = (int(gaze[0]), int(gaze[1]))
    if -1 not in center:
        cv2.circle(canvas, center, 10, (0, 255, 0), -1)
    return canvas

def draw_class(image, label):
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
    (X_combined_batch, Y_past_batch), _ = test_sample
    gaze_pred = model((X_combined_batch, Y_past_batch), training=False)
    predicted_gaze_norm = gaze_pred[0].numpy()  
    print("Predicted normalized gaze:", predicted_gaze_norm)
    predicted_gaze_norm = np.clip(predicted_gaze_norm, 0.0, 1.0)
    predicted_gaze = predicted_gaze_norm * (GAZE_MAX - GAZE_MIN) + GAZE_MIN
    return predicted_gaze

def show_videos():

    global last_predicted
    cap_world = cv2.VideoCapture("/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/NewLabeledVideo/world.mp4")
    labels = np.load("/Users/thaneshn/Desktop/ActionAnticipation-master/DataSet/NewLabeledVideo/action_labels.npy")
    
    test_iterator = iter(test_dataset)
    
    while True:
        ret, frame = cap_world.read()
        if not ret:
            break

        current_label = labels[0]
        gt_gaze = current_label[-2:]
        
        try:
            test_sample = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_dataset)
            test_sample = next(test_iterator)
        
        predicted_gaze = predict_gaze_from_test_sample(model, test_sample)
        
        frame_gt = draw_gaze(frame, gt_gaze)
        frame_pred = draw_gaze(frame, predicted_gaze)
        
        if last_predicted is not None:
            pt1 = (int(last_predicted[0]), int(last_predicted[1]))
            pt2 = (int(predicted_gaze[0]), int(predicted_gaze[1]))
            cv2.line(frame_pred, pt1, pt2, (0, 255, 255), thickness=2)
        
        last_predicted = predicted_gaze
        
        frame_concat = np.concatenate((frame_gt, frame_pred), axis=1)
        frame_concat = draw_class(frame_concat, current_label[1])
        cv2.imshow("Action Dataset", frame_concat)
        if cv2.waitKey(25) == 27: 
            break

    cap_world.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_videos()