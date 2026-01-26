MindCare AI: Mental Health Assistant using Machine Learning 🧠🤖

MindCare AI is an automated framework designed to analyze user text statements to identify mental health states. This system acts as a non-judgmental first point of contact, providing early detection and research-backed awareness.
📖Table of Contents
Introduction

Main Functionalities

Technical Methodology

How to Run

The Team & Task Distribution

1. Introduction
Individuals often feel more comfortable expressing internal struggles, such as anxiety or stress, through text than in face-to-face clinical settings. Trained on a dataset of over 53,000 human statements, MindCare AI learns linguistic patterns to distinguish between a "Normal" state and various clinical conditions.


2. Main Functionalities

Multi-Class Classification: Categorizes user input into 7 distinct mental health labels.


Risk Alert Mechanism: Triggers emergency guidance and displays helpline contacts when high-risk states like suicidal ideation are detected.


Prescriptive Engine: Maps predicted labels to specific supportive recommendations and therapeutic suggestions.



Interactive Web Interface: Provides a real-time chatbot experience built on Streamlit for seamless user interaction.


3. Technical Methodology
3.1 Dataset

Name: Sentiment Analysis on Mental Health Dataset.


Total Samples: 53,042 labeled statements.


Classes (7): Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, and Personality Disorder.

3.2 NLP Pipeline
The system converts raw text into a machine-readable format using several Natural Language Processing steps:


Cleaning: Removal of special characters, URLs, and punctuation.


Tokenization: Breaking text into individual words.


Lemmatization: Reducing words to their base form (e.g., "stresses" to "stress").


TF-IDF Vectorization: Assigning importance weights to emotional keywords.

3.3 AI Models

Classification: Primarily utilizes Support Vector Machine (SVM) and Logistic Regression for text classification.


Evaluation: Performance is measured using Accuracy, Precision, Recall, and Confusion Matrices.

4. How to Run
Install Dependencies:

Bash

pip install streamlit pandas scikit-learn nltk joblib
Launch the Application:

Bash

streamlit run app.py
5. The Team & Task Distribution
Based on our project implementation, the responsibilities were divided as follows:

Hafiza Yumna Mushtaq (243778): Data Preprocessing & Cleaning (NLTK implementation).

Rabia Anees (243746): Interface Development & Application Design (Streamlit).

Durr e Nayab (243735): Model Training & Performance Metrics Evaluation.

Suaiba Zainab (243751): Dataset Preparation, Data Splitting & System Testing.

Instructor: Saad-Ur-Rehman Program: BS Software Engineering

Would you like me to add a section for the "Emergency Helplines" that your app displays during the Suicidal Alert?
