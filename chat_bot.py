
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Training Data
train_texts = [
    "I am so excited and having a great day",
    "This was the best birthday surprise ever",
    "I am furious and completely mad at this",
    "Stop yelling at me right now",
    "I feel so lonely and heartbroken today",
    "I cannot stop crying and feeling down"
]

train_labels = [
    "Happy",
    "Happy",
    "Angry",
    "Angry",
    "Sad",
    "Sad"
]

# 2. Test Data
test_texts = [
    "I got the job and feel wonderful",
    "This makes my blood boil with rage",
    "I feel empty and depressed inside"
]

test_labels = ["Happy", "Angry", "Sad"]

#Emotion Responses
responses = {
    "Happy": "Happy to hear that! Keep smiling.",
    "Angry": "Stay calm. Take a deep breath.",
    "Sad": "I feel sorry for you. Everything will be alright."
}

# 3. Build the Pipeline (TF-IDF + Logistic Regression)
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True
    )),
    ('clf', LogisticRegression(random_state=42))
])

# 4. Train Model
text_clf.fit(train_texts, train_labels)

# 5. Test Predictions
predictions = text_clf.predict(test_texts)
probs = text_clf.predict_proba(test_texts)

print("--- Live Predictions ---\n")

for text, pred, prob in zip(test_texts, predictions, probs):
    confidence = np.max(prob) * 100

    print("Input:", text)
    print("Emotion:", pred)
    print("Response:", responses[pred])
    print(f"Confidence: {confidence:.2f}%")
    print()

# 6. Classification Report
print("--- Classification Report ---")
print(classification_report(
    test_labels,
    predictions,
    zero_division=0
))

# User Input Method
while True:
    user_input = input("Enter a sentence: ")

    if user_input == "exit":
        print("Thank you")
        break

    result = text_clf.predict([user_input])[0]

    print("Emotion:", result)
    print("Response:", responses[result])
    print()
