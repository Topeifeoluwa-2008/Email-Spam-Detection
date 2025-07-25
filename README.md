# Email Spam Detection
This repository contains a Jupyter notebook that demonstrates how to build and evaluate different machine learning models for email spam detection. The project utilizes common libraries like pandas, numpy, 
scikit-learn, and wordcloud to preprocess data, train models, and visualize results.

## Project Overview
The notebook covers the following steps:
### Dataset 
the dataset was collected from kaggle(Email_Spam.csv)

Data Loading and Exploration: Loading the email spam dataset and performing initial data exploration to understand its structure and content.

Text Preprocessing: Cleaning and preparing the email text data for model training, including handling stopwords.

Word Cloud Visualization: Generating word clouds to visualize the most frequent words in the email text.

Model Training and Evaluation: Training and evaluating different classification models for spam detection, including:
- Support Vector Machines (SVM)
- Naive Bayes
- Logistic Regression
- XGBoost
  
Model Comparison: Comparing the performance of the trained models using metrics like accuracy, precision, recall, and F1-score.

Model Saving: Saving the trained Naive Bayes model using pickle.
