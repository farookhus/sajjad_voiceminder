"""
train_model.py  —  How good is our category AI?  Run:  python train_model.py

It trains on data/training_sentences.csv and then gives the model a TEST
it has never seen (data/test_sentences.csv). Write the test sentences
yourself, and never copy them into the training file — that would be cheating!
"""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from brain import CONFIDENCE_THRESHOLD, train_classifier

train = pd.read_csv("data/training_sentences.csv")
test = pd.read_csv("data/test_sentences.csv")

print(f"Training sentences: {len(train)}")
print(train["category"].value_counts().to_string(), "\n")

model = train_classifier("data/training_sentences.csv")

predictions = model.predict(test["sentence"])
confidence = model.predict_proba(test["sentence"]).max(axis=1)
accuracy = accuracy_score(test["category"], predictions)
print(f"TEST ACCURACY: {accuracy:.0%}  ({(predictions == test['category']).sum()} of {len(test)} correct)\n")

labels = sorted(train["category"].unique())
matrix = pd.DataFrame(confusion_matrix(test["category"], predictions, labels=labels),
                      index=[f"real {l}" for l in labels], columns=labels)
print("CONFUSION MATRIX (rows = the real answer, columns = what the AI said):")
print(matrix.to_string(), "\n")

print("MISTAKES (learn from these — add similar sentences to the training file):")
for sentence, real, guess, conf in zip(test["sentence"], test["category"], predictions, confidence):
    if real != guess:
        print(f"  '{sentence}'  real={real}  AI said={guess} ({conf:.0%} sure)")

unsure = (confidence < CONFIDENCE_THRESHOLD).sum()
print(f"\nThe app would say 'Other' for {unsure} test sentence(s) because it was less than "
      f"{CONFIDENCE_THRESHOLD:.0%} sure.")
