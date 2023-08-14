import os
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from PIL import Image
import matplotlib.pyplot as plt

# Set working directory
address_saving = ""
address_stop_words = ""
address_code = ""
os.chdir(address_code)

# Load CSV
df = pd.read_csv("Mobile_Learning3500.csv")
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)

# Text cleaning
df['Tweet'] = df['text']
df['text'] = df['text'].str.replace(r'(<[^>]{0,}>|\r|\n)', ' ', regex=True)
df['text'] = df['text'].str.replace(r'_', '', regex=True)
df['text'] = df['text'].str.replace(r'[^\x01-\x7F]+', '', regex=True)
df['text'] = df['text'].str.replace(r'\W+', ' ', regex=True)
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace(r'\d+', '', regex=True)
df['text'] = df['text'].str.replace(r'([a-z])\1{3,}', r'\1', regex=True)

# Lemmatize or stem, remove stopwords, etc. (apply other text preprocessing steps)

# 3-gram
def create_ngram(text, n):
    words = text.split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return ngrams

df['trigrams'] = df['text'].apply(lambda x: create_ngram(x, 3))

# Save cleaned data
filename = f"CleanData_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
df.to_csv(os.path.join(address_saving, filename), index=False, encoding='utf-8')

# Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate_from_frequencies(Counter(" ".join(df['text']).split()))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

