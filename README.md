# Mental Health Text Analyzer

A deep learning NLP system that classifies mental health-related text into 7 categories and provides personalized wellness tips based on the detected condition.

---

## Overview

Mental health issues are often expressed through text — social media posts, journal entries, messages. This project uses a Bidirectional LSTM model trained on real-world mental health text data to detect patterns associated with Depression, Anxiety, Stress, Bipolar disorder, Personality disorder, Suicidal ideation, or Normal mental state and responds with actionable wellness tips.

---

## Demo

Type how you are feeling into the text box, click Analyze, and the model returns the detected mental health condition, model confidence score, and personalized wellness tips for that condition.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas and NumPy | Data loading, cleaning, null handling |
| NLTK | Text preprocessing, tokenization, lemmatization |
| TensorFlow and Keras | BiLSTM model training and inference |
| Scikit-learn | Label encoding, class weights, evaluation metrics |
| Matplotlib and Seaborn | EDA visualizations, confusion matrix |
| Streamlit | Interactive web application |

---

## Dataset

Sentiment Analysis for Mental Health from Kaggle containing 53,043 rows of social media text labeled across 7 mental health categories.

Link: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

### Class Distribution

| Class | Count |
|---|---|
| Normal | 16,351 |
| Depression | 15,404 |
| Suicidal | 10,653 |
| Anxiety | 3,888 |
| Bipolar | 2,877 |
| Stress | 2,669 |
| Personality disorder | 1,201 |

---

## Project Structure

| File | Description |
|---|---|
| app.py | Streamlit web application |
| preprocess.py | Text cleaning and tokenization pipeline |
| lstm_model.ipynb | Model training notebook |
| bidirectional_lstm.h5 | Saved trained model |
| Tokenizer.pkl | Saved tokenizer |
| LabelEncoder.pkl | Saved label encoder |
| requirements.txt | All dependencies |

---

## Model Architecture

| Layer | Details |
|---|---|
| Embedding | vocab 20000, dim 128, input length 150 |
| Bidirectional LSTM | 128 units, return sequences True |
| Dropout | 0.3 |
| Bidirectional LSTM | 64 units, return sequences False |
| Dropout | 0.2 |
| Dense | 64 units, ReLU |
| Dense | 32 units, ReLU |
| Dense | 7 units, Softmax |

- Loss: sparse categorical crossentropy
- Optimizer: Adam
- Callbacks: EarlyStopping with patience 3 and restore best weights True

---

## Text Preprocessing Pipeline

1. Lowercase conversion
2. Remove URLs, mentions and hashtags
3. Remove special characters and digits
4. Word tokenization using nltk word tokenize
5. Stopword removal with negation words kept (not, never, no, hardly)
6. Lemmatization using WordNetLemmatizer

Negation words are intentionally kept because "not okay" and "okay" carry completely different meaning in mental health context.

---

## Handling Imbalanced Data

The dataset has a 13x difference between the largest class Normal with 16,351 rows and the smallest class Personality disorder with 1,201 rows.

Solution used was compute class weight from scikit-learn with balanced mode. This assigns higher penalty to minority class misclassifications during training without adding any RAM overhead.

---

## Handling Missing Values

362 null values existed in the statement column concentrated heavily in minority classes. Personality disorder had 124 nulls out of 1,201 total rows which is 10 percent of that class. Dropping these rows was not viable so each null was filled with a randomly sampled real statement from the same class preserving label integrity and losing zero data.

---

## Results

| Metric | Value |
|---|---|
| Overall Accuracy | 67 percent |
| Strong classes | Normal, Anxiety |
| Challenging classes | Personality disorder, Depression |

Accuracy improves significantly with 15 to 20 epochs. Semantic overlap between Depression and Suicidal vocabulary is a known challenge in this dataset.
