# Gerekli kütüphaneleri içe aktaralım
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import re
from imblearn.over_sampling import SMOTE
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# nltk stopwords yükleyelim
nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))

# Veri setini yükleyelim
df = pd.read_csv('C:\\Users\\mehme\\OneDrive\\Masaüstü\\Meriç Yaman\\NLP\\proje\\dataset.csv')

# Veri setine göz atalım
print(df.head())

# 1. Veri Ön İşleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# 2. Özellik Çıkarımı
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Unigrams ve Bigrams
X = tfidf.fit_transform(df['cleaned_text']).toarray()
y = df['label']

# Veri setini ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE uygulayalım
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Lojistik Regresyon Modeli
logistic_model = LogisticRegression()
logistic_model.fit(X_train_resampled, y_train_resampled)
y_pred_logistic = logistic_model.predict(X_test)

print("Lojistik Regresyon Doğruluk Skoru:", accuracy_score(y_test, y_pred_logistic))
print("Lojistik Regresyon Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_logistic))

# Sinir Ağı Modeli
input_dim = X_train_resampled.shape[1]  # TF-IDF özellik boyutu

nn_model = Sequential([
    Dense(128, activation='relu', input_dim=input_dim),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')  # Çoklu sınıf için softmax kullanılıyor
])

nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitelim
nn_model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Test veri seti ile tahmin yapalım
y_pred_nn = np.argmax(nn_model.predict(X_test), axis=1)

print("Sinir Ağı Doğruluk Skoru:", accuracy_score(y_test, y_pred_nn))
print("Sinir Ağı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_nn))

# Kullanıcıdan cümle alıp tahmin yapma
label_dict = {
    0: "Suicidal Thoughts",
    1: "Eating Disorders",
    2: "Sleep Disorders",
    3: "Sexual Disorders",
    4: "Addictions",
    5: "Anger Control Disorders",
    6: "Borderline",
    7: "Psychosomatic Disorders",
    8: "OCD (Obsessive-Compulsive Disorder)",
    9: "Behavioral Disorders in Children",
    10: "Depression and Related Disorders",
    11: "Family and Relationship Issues",
    12: "Sports Psychology",
    13: "Attention Deficit and Hyperactivity Disorder (ADHD)",
    14: "Trauma",
    15: "Paraphilic Disorders"
}

def classify_sentences():
    sentences = []
    while True:
        print("\nSeçenekler:")
        print("1. Cümle Gir")
        print("2. Sonucu Gör")
        print("3. Çıkış")
        choice = input("Bir seçenek girin: ")

        if choice == '1':
            sentence = input("Lütfen bir cümle girin: ")
            sentences.append(sentence)
            print(f"Cümle eklendi: {sentence}")
        elif choice == '2':
            if not sentences:
                print("Lütfen önce cümle girin!")
                continue
            cleaned_sentences = [clean_text(sentence) for sentence in sentences]
            vectorized_sentences = tfidf.transform(cleaned_sentences).toarray()

            logistic_predictions = logistic_model.predict(vectorized_sentences)
            nn_predictions = np.argmax(nn_model.predict(vectorized_sentences), axis=1)

            print("\nSonuçlar:")
            for i, sentence in enumerate(sentences):
                logistic_class = label_dict.get(logistic_predictions[i], "Bilinmeyen Rahatsızlık")
                nn_class = label_dict.get(nn_predictions[i], "Bilinmeyen Rahatsızlık")
                print(f"Cümle {i+1}: '{sentence}'")
                print(f"  Lojistik Regresyon Tahmini: {logistic_predictions[i]} ({logistic_class})")
                print(f"  Sinir Ağı Tahmini: {nn_predictions[i]} ({nn_class})")

            sentences.clear()
        elif choice == '3':
            print("\u00c7ıkılıyor...")
            break
        else:
            print("Geçersiz seçenek. Lütfen tekrar deneyin.")

classify_sentences()
