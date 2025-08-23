# Sentiment Classification of Movie Reviews using Deep Learning Models

## Project Overview

This project focuses on binary sentiment classification of movie reviews using deep learning models. Given a dataset of positive and negative review texts, the goal was to design, train, and evaluate models that can accurately predict the sentiment of unseen reviews.

The project was completed as part of the DSCI 552 course (Machine Learning for Data Science) at the University of Southern California during Summer 2024.

## Objective

- Preprocess and vectorize raw text data for input into neural network models.
- Build and compare multiple deep learning architectures for binary classification.
- Evaluate model performance using standard classification metrics.
- Analyze model behavior and convergence during training.

## Models Implemented

- **Multilayer Perceptron (MLP)**  
  A basic feedforward network using word frequency-based input representations.

- **1D Convolutional Neural Network (CNN)**  
  Designed to extract local n-gram patterns in review sequences for sentiment classification.

- **Long Short-Term Memory (LSTM)**  
  Sequence modeling architecture used to capture long-range dependencies in review texts.

## Methodology

- **Data Preparation**

  - Tokenization and sequence padding
  - Word embedding (e.g., via Embedding layer or pretrained vectors)
  - Dataset split into training and validation sets

- **Model Training**

  - Trained each model using binary cross-entropy loss and the Adam optimizer.
  - Hyperparameters (e.g., batch size, learning rate) tuned manually.
  - Implemented dropout and early stopping for regularization.

- **Evaluation**
  - Accuracy, Precision, Recall, and F1 Score reported.
  - Training/validation loss and accuracy plotted per epoch.

## Summary of Findings

- CNN and LSTM models both significantly outperformed the baseline MLP, indicating the value of sequence-aware architectures.
- The LSTM model showed more stable convergence and better generalization on the validation set.
- Training performance was sensitive to embedding dimensionality and dropout rate.
- Preprocessing (e.g., text cleaning, sequence length cutoff) had a noticeable impact on performance consistency.

---

## Repository Structure

This repository is organized into the following structure:

### `data/`

Contains the raw text files used for training and evaluation.  
Includes:

- `pos/`: Directory of movie reviews with positive sentiment.
- `neg/`: Directory of movie reviews with negative sentiment.

### `notebook/`

Contains the main analysis and model training notebook.

- `Yoo_Paul_FP.ipynb`: Jupyter Notebook containing the full pipeline, including text preprocessing using Keras Tokenizer, implementation of MLP, 1D CNN, and LSTM models, training and evaluation using binary cross-entropy loss, and visualization of training accuracy and loss curves for each model.

---

## Author

**Paul Yoo**  
M.S. in Applied Data Science  
University of Southern California, Summer 2024  
Model implementation, training optimization, performance evaluation, and report documentation  
[LinkedIn](https://www.linkedin.com/in/pkyoo) | [GitHub](https://github.com/PKYOO-116)

---

### References

#### (b) Data Exploration and Pre-processing

[1] TensorFlow, “tf.keras.preprocessing.text.Tokenizer,” TensorFlow Documentation, 2024. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer. [Accessed: Jul. 24, 2024]. <br>
[2] TensorFlow, “tf.keras.preprocessing.sequence.pad_sequences,” TensorFlow Documentation, 2024. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences. [Accessed: Jul. 24, 2024]. <br>

#### (c) Word Embedding

[3] TensorFlow, “tf.keras.models.Sequential,” TensorFlow Documentation, 2024. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/models/Sequential. [Accessed: Jul. 24, 2024]. <br>
[4] TensorFlow, “tf.keras.layers.Input,” TensorFlow Documentation, 2024. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/Input. [Accessed: Jul. 24, 2024]. <br>
[5] TensorFlow, “tf.keras.layers.Embedding,” TensorFlow Documentation, 2024. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding. [Accessed: Jul. 24, 2024]. <br>
[6] TensorFlow, “tf.keras.layers.Flatten,” TensorFlow Documentation, 2024. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten. [Accessed: Jul. 24, 2024]. <br>
[7] Real Python, “Python Keras Text Classification,” [Online]. Available: https://realpython.com/python-keras-text-classification/. [Accessed: Jul. 24, 2024]. <br>
[8] Kaggle, “A Detailed Explanation of Keras Embedding Layer,” [Online]. Available: https://www.kaggle.com/code/rajmehra03/a-detailed-explanation-of-keras-embedding-layer. [Accessed: Jul. 24, 2024]. <br>

#### (d) Multi-Layer Perceptron

[9] TensorFlow, “tf.keras.layers.Dense,” TensorFlow Documentation, 2024. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense. [Accessed: Jul. 24, 2024]. <br>
[10] TensorFlow, “tf.keras.layers.Dropout,” TensorFlow Documentation, 2024. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout. [Accessed: Jul. 24, 2024]. <br>
[11] J. Brownlee, “Build Multi-Layer Perceptron Neural Network Models with Keras,” Machine Learning Mastery, [Online]. Available: https://machinelearningmastery.com/build-multi-layer-perceptron-neural-network-models-keras/. [Accessed: Jul. 24, 2024]. <br>
[12] DataCamp, “Multilayer Perceptrons in Machine Learning,” [Online]. Available: https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning. [Accessed: Jul. 24, 2024]. <br>
[13] Towards Data Science, “Multilayer Perceptron Explained with a Real-Life Example and Python Code,” [Online]. Available: https://towardsdatascience.com/multilayer-perceptron-explained-with-a-real-life-example-and-python-code-sentiment-analysis-cb408ee93141. [Accessed: Jul. 24, 2024]. <br>
[14] ScienceDirect, “Multilayer Perceptron,” [Online]. Available: https://www.sciencedirect.com/topics/computer-science/multilayer-perceptron. [Accessed: Jul. 24, 2024]. <br>

#### (e) One-Dimensional Convolutional Neural Network

[15] TensorFlow, “tf.keras.layers.Conv1D,” TensorFlow Documentation, 2024. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D. [Accessed: Jul. 24, 2024]. <br>
[16] TensorFlow, “tf.keras.layers.MaxPool1D,” TensorFlow Documentation, 2024. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool1D. [Accessed: Jul. 24, 2024]. <br>
[17] GeeksforGeeks, “What is a 1D Convolutional Layer in Deep Learning?” [Online]. Available: https://www.geeksforgeeks.org/what-is-a-1d-convolutional-layer-in-deep-learning/. [Accessed: Jul. 24, 2024]. <br>
[18] ScienceDirect, “A Novel Neural Network Model for Time Series Forecasting,” [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0888327020307846. [Accessed: Jul. 24, 2024]. <br>

### (f) Long Short-Term Memory Recurrent Neural Network

[19] TensorFlow, “tf.keras.layers.LSTM,” TensorFlow Documentation, 2024. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM. [Accessed: Jul. 24, 2024]. <br>
[20] TensorFlow, “Working with RNNs,” TensorFlow Documentation, 2024. [Online]. Available: https://www.tensorflow.org/guide/keras/working_with_rnns. [Accessed: Jul. 24, 2024]. <br>
[21] Turing, “Recurrent Neural Networks and LSTM,” [Online]. Available: https://www.turing.com/kb/recurrent-neural-networks-and-lstm. [Accessed: Jul. 24, 2024]. <br>
[22] A. Mittal, “Understanding RNN and LSTM,” Medium, [Online]. Available: https://aditi-mittal.medium.com/understanding-rnn-and-lstm-f7cdf6dfc14e. [Accessed: Jul. 24, 2024]. <br>
