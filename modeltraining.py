import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load the processed dataset
# Ensure the filename matches your local file
df = pd.read_csv('Processed_Mental_Health_Data.csv')

# 2. Data Partitioning (80% Training, 20% Testing)
# Standard split to evaluate model performance on unseen data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the split datasets for documentation/viva purposes
train_df.to_csv('Training_Data_80.csv', index=False)
test_df.to_csv('Testing_Data_20.csv', index=False)

X_train = train_df['statement']
y_train = train_df['status']
X_test = test_df['statement']
y_test = test_df['status']

# 3. Text Vectorization using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. Model Training: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test_tfidf)

# --- PERFORMANCE METRICS OUTPUT ---
print("--- MODEL PERFORMANCE REPORT ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nDetailed Classification Report:\n", classification_report(y_test, y_pred))

# 6. Confusion Matrix Visualization
# Useful for identifying misclassifications between categories
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y_test.unique())

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('MindCare AI - Confusion Matrix (Model Error Analysis)')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')

# Save and display the visualization
plt.savefig('confusion_matrix.png')
print("\nConfusion Matrix graph saved as 'confusion_matrix.png'")
plt.show()

# 7. Serialize Model and Vectorizer for Deployment
joblib.dump(model, 'mental_health_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("\nTraining Complete! Model and Vectorizer have been successfully saved.")