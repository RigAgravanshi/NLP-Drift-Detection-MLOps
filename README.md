# NLP Drift Detection MLOps

A short, description:

We will train an HF transformer on Banking77 dataset from HF(originally we were trying on Customer tickets, of certain issues based on Type and Urgency).
Our main objective is MLOps, i.e. we will deploy, train and track it using FastAPI, Streamlit, MLFlow frameworks.
Then we will simulate data and analyze behaviour "Post-Deployment". This demonstrates Drift Detection and explains ML model lifecycle awareness.

Intent is:
A semantic property of the text

Not employing TF-IDF and other such stuff. We focus on deployment here , so we didnt want to spend time in NLP techniques(stopwords removal, lemmatization and such)

DistilBERT is perfect as a second-stage model, but not for this time
