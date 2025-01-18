import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud 
url = "https://lazyprogrammer.ne/course files/spam.csv"  # Update with the correct URL
df = pd.read_csv(url, encoding="ISO-8859-1")

# Drop unnecessary columns
columns_to_drop = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]
df = df.drop(columns=columns_to_drop, axis=1) 

df['b_labels'] = df['v1'].map({'ham': 0, 'spam': 1}) 
X_train, X_test, y_train, y_test = train_test_split(messages, binary_labels, test_size=0.33, random_state=42)

# Create a CountVectorizer with specific options
featurizer = CountVectorizer(decode_error='ignore')  # Add more options as needed

# Fit and transform the training data
X_train_features = featurizer.fit_transform(X_train)

# Transform the test data
X_test_features = featurizer.transform(X_test) 
model = MultinomialNB()
model.fit(X_train_features, y_train)

# Evaluate the model on training data
train_accuracy = model.score(X_train_features, y_train)
print("Train accuracy:", train_accuracy)

# Evaluate the model on test data
test_accuracy = model.score(X_test_features, y_test)
print("Test accuracy:", test_accuracy)

# Make predictions
y_train_pred = model.predict(X_train_features)
y_test_pred = model.predict(X_test_features)

# Calculate F1-score
train_f1 = f1_score(y_train, y_train_pred)
print("Train F1-score:", train_f1)

test_f1 = f1_score(y_test, y_test_pred)
print("Test F1-score:", test_f1)  
# Calculate probabilities
prob_train = model.predict_proba(X_train_features)[:, 1]
prob_test = model.predict_proba(X_test_features)[:, 1]

# Calculate AUC-ROC scores
train_auc = roc_auc_score(y_train, prob_train)
print("Train AUC-ROC score:", train_auc)

test_auc = roc_auc_score(y_test, prob_test)
print("Test AUC-ROC score:", test_auc)

# Confusion matrix
confusion_mat_train = confusion_matrix(y_train, y_train_pred)
confusion_mat_test = confusion_matrix(y_test, y_test_pred)

print("Confusion matrix (Train):\n", confusion_mat_train)
print("Confusion matrix (Test):\n", confusion_mat_test) 
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

print("Train Accuracy:", train_accuracy)
print("Train Precision:", train_precision)
print("Train Recall:", train_recall)

print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)

# Calculate probabilities
prob_train = model.predict_proba(X_train_features)[:, 1]
prob_test = model.predict_proba(X_test_features)[:, 1]

# Calculate AUC-ROC scores
train_auc = roc_auc_score(y_train, prob_train)
print("Train AUC-ROC score:", train_auc)

test_auc = roc_auc_score(y_test, prob_test)
print("Test AUC-ROC score:", test_auc)

# Confusion matrix
confusion_mat_train = confusion_matrix(y_train, y_train_pred)
confusion_mat_test = confusion_matrix(y_test, y_test_pred)

print("Confusion matrix (Train):\n", confusion_mat_train)
print("Confusion matrix (Test):\n", confusion_mat_test)







