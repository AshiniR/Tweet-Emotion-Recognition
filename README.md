# 📝 Tweet Emotion Recognition with TensorFlow

This project builds a deep learning model that can recognize emotions in short tweets.  
It is based on the **Tweet Emotion Dataset** from Hugging Face.

---

## 📌 Overview
The goal of this project is to classify tweets into different emotions such as:
- Joy
- Sadness
- Anger
- Love
- Fear
- Surprise

We use **Natural Language Processing (NLP)** with TensorFlow to train a model that understands text and predicts the correct emotion.

---

## 🚀 Steps in the Notebook

### 1. Introduction
- Learn what the project is about and what we are trying to solve.

### 2. Setup & Imports
- Install and import the required libraries like TensorFlow, NumPy, matplotlib, and Hugging Face datasets.

### 3. Importing Data
- Load the **Tweet Emotion Dataset**.
- Split it into training, validation, and test sets.

### 4. Tokenizer
- Convert words into numbers (sequences) so the model can understand them.

### 5. Padding & Truncating
- Make all tweet sequences the same length (short ones padded, long ones cut).

### 6. Preparing Labels
- Convert text labels (e.g., “joy”) into numbers for training.

### 7. Building the Model
- Use **Embedding layers** to represent words.
- Use **Bidirectional LSTM layers** to capture meaning and context.
- Add a **Dense layer** with softmax to output probabilities for each emotion.

### 8. Training
- Train the model with training data and check performance on validation data.

### 9. Evaluation
- Plot accuracy and loss graphs.
- Test the model on unseen tweets.
- Show a confusion matrix to understand mistakes.

---

## 🛠 Requirements
Make sure you have the following installed:
- Python 3.8+
- TensorFlow
- Hugging Face Datasets
- NumPy
- Matplotlib
- Scikit-learn

You can install them using:
```bash
pip install tensorflow datasets numpy matplotlib scikit-learn
```
## 📊 Example Output
After training, the model can take a new tweet and predict the emotion.
Example:

Sentence: I just got a new job today!
Emotion: joy
Predicted Emotion: joy
## 🎯 Learning Outcomes
By going through this notebook, you will learn:

* How to preprocess text for NLP tasks.
* How to build and train an LSTM model for emotion recognition.
* How to evaluate NLP models using accuracy, loss, and confusion matrices.

## 📚 References
- [Tweet Emotion Dataset](https://github.com/dair-ai/emotion_dataset)  
- [TensorFlow Documentation](https://www.tensorflow.org/)  
- [Coursera Guided Project: Tweet Emotion Recognition](https://www.coursera.org/projects/tweet-emotion-tensorflow)  
