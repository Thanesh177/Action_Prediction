# Action Anticipation Model

This project implements a gaze and action prediction framework using deep learning and temporal modeling techniques. The system leverages head motion data, joint and body features, and gaze vectors to predict future actions and gaze directions in a sequence-to-one classification/regression setting.


To set up the environment for this gaze and action prediction framework, install the following dependencies:


Python >= 3.8


numpy>=1.20        # Numerical computations
pandas>=1.3        # DataFrame and CSV support
scipy>=1.7         # .mat file loading and scientific operations
matplotlib>=3.4    # Plotting and visualizations
scikit-learn>=1.0  # KMeans, train_test_split, scaling
tensorflow>=2.11   # Deep learning (LSTM, attention, mixed precision, JIT)
opencv-python>=4.5  # For video I/O and gaze overlay
ruptures>=1.1      # For CPD downsampling
numpy>=1.20
pandas>=1.3
scipy>=1.7
matplotlib>=3.4
scikit-learn>=1.0
tensorflow>=2.11
opencv-python>=4.5
ruptures>=1.1



Should use TensorFlow >= 2.11 to ensure compatibility with mixed precision and JIT compilation.

If using a GPU, install the appropriate version of tensorflow-gpu with CUDA 11.2 or higher.

Dataset:

1. Dataset1 (ANTICIPATE): https://vislab.isr.tecnico.ulisboa.pt/datasets_and_resources/#HRIcups

2. Dataset2 (Ball_catch): https://drive.google.com/drive/folders/15J5jZx1aFi-WjWJSHR86sFClNQgpH75G

Dataset.py - Contains core data loading functions such as reconstructStructure(), create_reference(), and load(). It handles the extraction and preprocessing of joint, body, and gaze features from .mat datasets for both action and gaze prediction tasks.


Action_prediction_dataset1.py - Implements an action prediction model using joint and body features with BiLSTM encoder-decoder architecture and attention mechanism. Uses temporal sequences and categorical cross-entropy for supervised classification.

Action_prediction_dataset2.py - Variation of the action prediction model using head velocity and head direction features from the ANTICIPATE dataset. Includes change point detection (CPD) for downsampling and uses BiLSTM + attention for robust temporal modeling.

Gaze_prediction.py - 	Predicts future gaze coordinates (x, y, z) using joint, body, and gaze features. The model employs a Bidirectional LSTM encoder with attention and a sequential decoder to output predicted gaze vectors.

vid.py - Visualization of the Gaze prediction also requires the output of Gaze_prediction.py to visualize results.


## Model Variant Comparison

| Variant                        | Final Validation Accuracy |
|-------------------------------|----------------------------|
| Clustered Joint + Body + Gaze | 92.4%                      |
| Gaze + Joint + Body (No Clust)| 92.7%                      |
| Clustered Joint + Body        | 86.8%                      |
| Joint + Body                  | 79.6%                      |

