# 📝 Tweet Emotion Recognition

 DAIR AI Emotion Dataset 
# 🧠 Introduction
Understanding emotions expressed in text is a key challenge in Natural Language Processing (NLP). This project focuses on Tweet Emotion Recognition — automatically classifying tweets into six emotion categories using a variety of models ranging from traditional machine learning to advanced transformer-based architectures.

* The project experiments with multiple models:
* SVM (Support Vector Machine)
* BLSTM (Bidirectional LSTM)
* DistilBERT
* BERT
* RoBERTa

The objective is to compare their performance on emotion detection tasks and identify the most effective model for real-world applications.

# 🎯Objectives

* Preprocess and clean tweet text data (remove mentions, links, hashtags, and emojis).
* Train and evaluate multiple models for emotion classification.
* Fine-tune transformer-based models (DistilBERT, BERT, RoBERTa) for multi-class emotion recognition.
* Address class imbalance using weighted loss functions and weighted metrics.
* Compare model performance using accuracy, precision, recall, and F1-score.
* Save the best-performing models for deployment.

# 📊 Dataset

Source: Tweet Emotion Recognition Dataset (Kaggle)
This dataset consists of tweets labeled with six emotions.

* 0	- Sadness	
* 1	- Joy	
* 2	- Love	
* 3	- Anger	
* 4	- Fear	
* 5	- Surprise

# 🧹 Preprocessing Steps

* Convert text to lowercase (SVM)
* Remove URLs, mentions (@username), and hashtags(SVM)
* Tokenize using the respective tokenizer (BERT / RoBERTa / DistilBERT)
* Apply dynamic padding and truncation
* Encode labels numerically
* Handle class imbalance using computed class weights

⚙️ Evaluation Metrics

Each model was evaluated using the following weighted metrics to account for class imbalance:

* Accuracy
* Weighted Precision
* Weighted Recall
* Weighted F1-Score

# 📁 Project Structure

```plaintext
tweet-emotion-recognition/
│
├── Data/
│   └──              
│
├── notebooks/
│   └── tweet-emotion-recognition.ipynb  
│
├── README.md                         
├── .gitignore
```
# Key Technologies
🤗 Transformers: Hugging Face transformers library
🔥 PyTorch: Deep learning framework (BERT, RoBERTa, DistilBERT)
💻 TensorFlow / Keras: Deep learning framework (BLSTM)
📊 Scikit-learn: Evaluation metrics, preprocessing, Traditional ML models (SVM) and feature extraction
🎯 Optuna: Hyperparameter optimization
📈 Matplotlib/Seaborn: Data visualization
🐼 Pandas: Data manipulation

# 🏆 Results and Comparison

| Model       | Accuracy | Precision (W) | Recall (W) | F1-Score (W) |
| ----------- | -------- | ------------- | ---------- | ------------ |
| SVM         | 0.85     | 0.84          | 0.85       | 0.84         |
| BLSTM       | 0.88     | 0.87          | 0.88       | 0.87         |
| DistilBERT  | 0.91     | 0.91          | 0.91       | 0.91         |
| BERT        | 0.93     | 0.93          | 0.93       | 0.93         |
| **RoBERTa** | **0.95** | **0.95**      | **0.95**   | **0.95**     |
