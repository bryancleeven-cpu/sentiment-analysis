#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-22T16:41:45.362Z
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

# Judul Aplikasi
st.title("Sentiment Analysis Model Evaluation")

# Upload Dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load Data
    data = pd.read_csv(uploaded_file, delimiter=';')
    
    # Tampilkan info dataset
    st.write(data.info())
    st.write(data.head())

    # Input untuk label yang digunakan
    label_names = ['Negatif', 'Netral', 'Positif']
    
    # Tampilkan Data Preprocessing
    st.header("Data Preprocessing")
    
    # Split data untuk training dan testing (80:20, 60:40, 50:50)
    splits = [
        ("80:20", train_test_split(data['stemmed_text'], data['Sentiment'], test_size=0.2, random_state=42, stratify=data['Sentiment'])),
        ("60:40", train_test_split(data['stemmed_text'], data['Sentiment'], test_size=0.4, random_state=42, stratify=data['Sentiment'])),
        ("50:50", train_test_split(data['stemmed_text'], data['Sentiment'], test_size=0.5, random_state=42, stratify=data['Sentiment']))
    ]

    # Inisialisasi vektorisasi
    vectorizer = CountVectorizer()

    # Placeholder untuk menyimpan hasil
    results = {}

    # Melakukan pelatihan dan evaluasi untuk setiap split (80:20, 60:40, 50:50)
    for ratio, (X_train, y_train, X_test, y_test) in splits:
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # SVM
        svm = SVC(kernel='linear', probability=True)
        svm.fit(X_train_vec, y_train)
        y_pred_svm = svm.predict(X_test_vec)
        cm_svm = confusion_matrix(y_test, y_pred_svm, labels=label_names)
        results[f"SVM - {ratio}"] = classification_report(y_test, y_pred_svm, target_names=label_names, output_dict=True)
        
        # Naive Bayes
        nb = MultinomialNB()
        nb.fit(X_train_vec, y_train)
        y_pred_nb = nb.predict(X_test_vec)
        cm_nb = confusion_matrix(y_test, y_pred_nb, labels=label_names)
        results[f"Naive Bayes - {ratio}"] = classification_report(y_test, y_pred_nb, target_names=label_names, output_dict=True)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf.fit(X_train_vec, y_train)
        y_pred_rf = rf.predict(X_test_vec)
        cm_rf = confusion_matrix(y_test, y_pred_rf, labels=label_names)
        results[f"Random Forest - {ratio}"] = classification_report(y_test, y_pred_rf, target_names=label_names, output_dict=True)

    # Menampilkan Classification Report dan Confusion Matrix
    for model_name, report in results.items():
        st.subheader(f"Classification Report: {model_name}")
        report_df = pd.DataFrame(report).transpose()  # Mengubah dict menjadi DataFrame
        st.dataframe(report_df.style.background_gradient(cmap="coolwarm").format(precision=3))
        
        # Plot Confusion Matrix untuk setiap model
        cm = cm_svm if "SVM" in model_name else (cm_nb if "Naive Bayes" in model_name else cm_rf)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix for {model_name}')
        st.pyplot(fig)