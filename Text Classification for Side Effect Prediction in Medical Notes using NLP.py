#!/usr/bin/env python
# coding: utf-8

# Text Classification for Side Effect Prediction in Medical Notes using NLP

# This Natural language algorithm builds a text classification pipeline by extracting data from a word document, including "Patient_ID," "Medication," "Doctor_Notes," and "Reported_Side_Effects." Labels are created based on the side effects, with 'Negative' for any side effect other than "None" and 'Positive' for "None." The text is transformed. Then, the dataset is spitted into training and testing sets, with a Logistic Regression model trained using class balancing. The model’s performance is evaluated using metrics like precision, recall, F1-score, and accuracy, along with a confusion matrix.

# In[1]:


# Install library
get_ipython().system('pip install python-docx')


# Load Libraries

# In[2]:


import os
from docx import Document
import pandas as pd
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Change directory to where the word document can be found

# In[3]:


os.chdir('C:\\Users\\joe62\\Downloads\\NLP')


# Load the word document and put in a dataframe

# In[4]:


# Load the Word document
doc = Document("NLP for polypharmacy and Patient safety.docx")

# Initialize list to store extracted data
data = []

# Loop through paragraphs in the document
for para in doc.paragraphs:
    text = para.text.strip()  # Strip leading and trailing whitespace
    if text:  # If there's any text in the paragraph
        print(f"Raw paragraph text: '{text}'")  # Print raw text for debugging
        print("-" * 50)  # Separator for better readability

        # Clean the text by removing any trailing commas
        if text.endswith(','):
            text = text[:-1]  # Remove trailing comma

        try:
            # Try to load the string as a JSON object
            note_data = json.loads(text)
            
            # Extract patient info from the parsed JSON
            patient_id = note_data.get("Patient_ID", "")
            medication = note_data.get("Medication", "")
            doctor_notes = note_data.get("Doctor_Notes", "")
            reported_side_effects = note_data.get("Reported_Side_Effects", "")
            
            # Append to the data list
            data.append({
                "Patient_ID": patient_id,
                "Medication": medication,
                "Doctor_Notes": doctor_notes,
                "Reported_Side_Effects": reported_side_effects
            })
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")  # Print the specific error message
            continue

# Convert extracted data to a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)


# In[5]:


# Save to CSV
df.to_csv("side_effects_dataset_.csv", index=False)

print("side_effects_dataset_.csv")


# Perform Text Classification using Natural language Processing

# In[6]:


# Convert extracted data to a DataFrame
df = pd.DataFrame(data)

# Boostrapped
df = df.sample(n=300, replace=True, random_state=42)


# In[7]:


# Example: Assume we want to classify Doctor_Notes into categories (e.g., 'Positive', 'Negative', 'Neutral')
# This classification could be based on the presence of certain keywords or sentiments in the doctor's notes.

# Create a mock 'Label' column (You can replace this with your actual labels)
# Here, we assume a simple classification based on side effects
df['Label'] = df['Reported_Side_Effects'].apply(lambda x: 'Negative' if x != 'None' else 'Positive')

# Prepare features and labels
X = df['Doctor_Notes']  # Text data
y = df['Label']  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a classifier (Logistic Regression in this case)
classifier = LogisticRegression()
classifier.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = classifier.predict(X_test_tfidf)

# Print classification report
print(classification_report(y_test, y_pred))

