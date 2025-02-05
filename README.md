# Text-Classification-for-Side-Effect-Prediction-in-Medical-Notes-using-NLP
This Natural language algorithm builds a text classification pipeline by extracting data from a word document,containing "Patient_ID," "Medication," "Doctor_Notes," and "Reported_Side_Effects." Labels are created based on the side effects, with 'Negative' for any side effect other than "None" and 'Positive' for "None." The text is transformed. Then, the dataset is spitted into training and testing sets, with a Logistic Regression model trained using class balancing. The model’s performance is evaluated using metrics like precision, recall, F1-score, and accuracy, along with a confusion matrix.


