import pandas as pd
import os
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

data_path = "bbc-news-data.csv"

df = pd.read_csv(data_path, delimiter="\t")
df.head()

# İlk birkaç satırı görüntüleyelim
#print(df.head())

# Veri setindeki sütunları listeleyelim
#print(df.columns)

# Eksik veri var mı kontrol edelim
#print(df.isnull().sum())

# Kategorilerin dağılımını görelim
#print(df['category'].value_counts())


# %% Metin Temizleme

def clean_text(text):
    text = text.lower() # Küçük harfe çevir
    text = re.sub(r'\W', '', text) # Özel karakterleri kaldır
    text = re.sub(r'\s+', '', text).strip() # Fazla boşlukları temizle
    return text

# 'content' sütununu temizleyelim
df['clean_content'] = df['content'].apply(clean_text)

print(df[['content', 'clean_content']].head())

# Kelime Köklerini Bulma (Stemming & Lemmatization)
#Stemming: Kelimelerin son eklerini keserek köküne indirger. (örneğin: running → run)
#Lemmatization: Kelimeleri sözlükteki kök haline çevirir. (örneğin: better → good)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Stemming ve Lemmatization fonksiyonları
def stem_text(text):
    words = word_tokenize(text)
    return ' '.join([stemmer.stem(word) for word in words])

def lemmatize_text(text):
    words = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in words])


df['stemmed_content'] = df['clean_content'].apply(stem_text)
df['lemmatized_content'] = df['clean_content'].apply(lemmatize_text)

print(df[['clean_content', 'stemmed_content', 'lemmatized_content']].head())







