# 📝 Tweet Emotion Recognition

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

Source: Tweet Emotion Recognition Dataset (Kaggle)( DAIR AI Emotion Dataset )
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

# ⚙️ Evaluation Metrics

Each model was evaluated using the following weighted metrics to account for class imbalance:

* Accuracy
* Weighted Precision
* Weighted Recall
* Weighted F1-Score

# 📁 Project Structure

```plaintext
tweet-emotion-recognition/
│
├── Data/emotions   
│   └── test.txt
|             
│
├── notebooks/
│   └── tweet-emotion-recognition.ipynb  
│
├── README.md                         
├── .gitignore
```
# Key Technologies
* 🤗 Transformers: Hugging Face transformers library
* 🔥 PyTorch: Deep learning framework (BERT, RoBERTa, DistilBERT)
* 💻 TensorFlow / Keras: Deep learning framework (BLSTM)
* 📊 Scikit-learn: Evaluation metrics, preprocessing, Traditional ML models (SVM) and feature extraction
* 🎯 Optuna: Hyperparameter optimization
* 📈 Matplotlib/Seaborn: Data visualization
* 🐼 Pandas: Data manipulation

# 🏆 Results and Comparison

| Model      | Accuracy | F1 (Weighted) | Precision (Weighted) | Recall (Weighted) |
| ---------- | -------- | ------------- | -------------------- | ----------------- |
| SVM        | 0.8915   | 0.8947        | 0.9040               | 0.8915            |
| BLSTM      | 0.9245   | 0.9219        | 0.9260               | 0.9245            |
| DistilBERT | 0.9300   | 0.9308        | 0.9328               | 0.9300            |
| BERT       | 0.9295   | 0.9309        | 0.9356               | 0.9295            |
| RoBERTa    | 0.9220   | 0.9239        | 0.9313               | 0.9220            |

Based on the evaluation metrics, DistilBERT is the best-performing model for this task. It achieves the highest overall accuracy (0.9300) and maintains strong weighted F1 (0.9308), precision (0.9328), and recall (0.9300), providing the best balance between all metrics.
