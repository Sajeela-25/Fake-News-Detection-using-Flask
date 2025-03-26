#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download necessary NLTK data
nltk.download('stopwords')

# Initialize stemming
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters and numbers
    words = text.split()  # Tokenization (split into words)
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords & apply stemming
    return ' '.join(words)

# Load dataset
df = pd.read_csv("news.csv")

# Apply text preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Split data
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
model_path = "fake_news_model.pkl"
with open(model_path, 'wb') as file:
    pickle.dump((vectorizer, model), file)

print(f"Model saved successfully at {model_path}")


# In[ ]:




