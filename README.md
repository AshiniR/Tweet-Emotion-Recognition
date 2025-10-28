# 📝 Tweet Emotion Recognition

## 🧠 Introduction
Understanding emotions expressed in text is a key challenge in Natural Language Processing (NLP). This project focuses on Tweet Emotion Recognition — classifying tweets into six emotion categories using a variety of models ranging from traditional machine learning to advanced transformer-based architectures.

The project experiments with multiple models:
* SVM (Support Vector Machine)
* BLSTM (Bidirectional LSTM)
* DistilBERT
* BERT
* RoBERTa

The objective is to compare their performance on emotion detection tasks and identify the most effective model for real-world applications.

## 🎯Objectives

* Preprocess text data specific to each model to optimize performance.
* Address class imbalance using weighted loss functions and weighted evaluation metrics.
* Train and evaluate multiple models for emotion classification using the Tweet Emotion Recognition dataset.
* Tune each model's hyperparameters using a validation set for better performance.
* Compare model performance using accuracy, precision, recall, and F1-score, evaluated on an unseen test set.

## 📊 Dataset

Source: [Tweet Emotion Recognition Dataset (Hugging Face – DAIR AI Emotion Dataset)](https://huggingface.co/datasets/dair-ai/emotion)

This dataset consists of tweets labeled with six emotions.

* 0	- Sadness	
* 1	- Joy	
* 2	- Love	
* 3	- Anger	
* 4	- Fear	
* 5	- Surprise

## Key Technologies Used

* 🤗 Transformers: Hugging Face transformers library
* 🔥 PyTorch: Deep learning framework (BERT, RoBERTa, DistilBERT)
* 💻 TensorFlow / Keras: Deep learning framework (BLSTM)
* 📊 Scikit-learn: Evaluation metrics, Hyperparameter optimization, Preprocessing, Traditional ML models (SVM)
* 🎯 Optuna: Hyperparameter optimization
* 📈 Matplotlib/Seaborn: Data visualization
* 🐼 Pandas: Data manipulation

## 🏆 Results and Comparison

Evaluation was done with an unseen test set.

| Model      | Accuracy | F1 (Weighted) | Precision (Weighted) | Recall (Weighted) |
| ---------- | -------- | ------------- | -------------------- | ----------------- |
| SVM        | 0.8915   | 0.8947        | 0.9040               | 0.8915            |
| BLSTM      | 0.9245   | 0.9219        | 0.9260               | 0.9245            |
| DistilBERT | 0.9300   | 0.9308        | 0.9328               | 0.9300            |
| BERT       | 0.9295   | 0.9309        | 0.9356               | 0.9295            |
| RoBERTa    | 0.9220   | 0.9239        | 0.9313               | 0.9220            |

Based on the evaluation metrics, **DistilBERT** is the best-performing model for this task. It achieves the highest overall accuracy (0.9300) and maintains strong weighted F1 (0.9308), precision (0.9328), and recall (0.9300), providing the best balance between all metrics.
