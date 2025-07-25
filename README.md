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

```
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```

<img width="794" height="407" alt="bticl_wordcloud" src="https://github.com/user-attachments/assets/042ed554-e4fa-4e97-a0ea-92f5fe12b7f3" />

```
email_text = ' '.join(df['text'].astype(str))

# Add more words to ignore
stopwords.update(["Customer service", "information", "may", "Team"])

# redo stopwords, limit number of words
wordcloud = WordCloud(stopwords=stopwords, max_words=25,
                      background_color="azure").generate(email_text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```

<img width="515" height="268" alt="bticl_wordcloud1" src="https://github.com/user-attachments/assets/ffbde70d-0e68-425c-8ad9-7c798e52a080" />

```
import matplotlib.pyplot as plt
# Generate the word cloud
wordcloud = WordCloud(
    stopwords=stopwords,
    mask=alice_mask,
    background_color="azure"
).generate(email_text)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

```

<img width="647" height="790" alt="bticl_wordcloud2" src="https://github.com/user-attachments/assets/0469d89c-14b0-4d29-9859-d4d160ea6f4e" />

```
from wordcloud import WordCloud, STOPWORDS
from os import path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Get the current working directory
d = path.dirname(__file__) if "__file__" in locals() else path.abspath(".")

# Define the path to your mask image
mask_path = path.join(d, "wordcloud.jpg")

# Read the mask image
alice_mask = np.array(Image.open(mask_path))

# Prepare stopwords
stopwords = set(STOPWORDS)
stopwords.update(["Customer service", "information", "may", "Team"])

# Provide some sample text - Using the email_text created earlier
email_text = ' '.join(df['text'].astype(str))


# Generate the word cloud
wordcloud = WordCloud(
    stopwords=stopwords,
    mask=alice_mask,
    max_words=25,
    background_color="azure"
).generate(email_text)

# Display the word cloud
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Save to file
wordcloud.to_file(path.join(d, "wordcloud.png"))
```

<img width="321" height="389" alt="bticl_wordcloud4" src="https://github.com/user-attachments/assets/331913bb-cf5a-4186-b153-96c8cbcbb51e" />

Model Training and Evaluation: Training and evaluating different classification models for spam detection, including:
- Support Vector Machines (SVM)
- Naive Bayes
- Logistic Regression
- XGBoost
  
Model Comparison: Comparing the performance of the trained models using metrics like accuracy, precision, recall, and F1-score.

Model Saving: Saving the trained Naive Bayes model using pickle.
