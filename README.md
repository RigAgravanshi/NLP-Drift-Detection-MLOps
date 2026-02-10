# NLP Drift Detection MLOps

A short, non-GPT description:

We will train an HF transformer on Customer tickets, of certain issues based on Type and Urgency.
Our main objective is MLOps, i.e. we will deploy,train and track it using FastAPI, Streamlit, MLFlow frameworks.
Then we will simulate data and analyze behaviour "Post-Deployment". This demonstrates Drift Detection and explains ML model lifecycle awareness.

“Each ticket has two independent labels, intent and urgency. I treat this as a multi-task learning problem where the same text input is used to predict both targets, and I perform a single stratified split to preserve intent distribution while keeping the labels aligned."
