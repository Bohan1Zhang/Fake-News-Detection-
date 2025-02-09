# Fake-News-Detection-
![Fake News](./png_file/news.png)
## Description of the project
The rapid spread of misinformation has become a pressing issue in the digital age, particularly with the rise of social media and online news platforms. Fake news can have serious consequences, from political manipulation to public misinformation about health crises. The goal of this project is to develop a machine learning model capable of distinguishing between real and fake news articles. By leveraging **BERT / DistilBERT**, a state-of-the-art NLP model, we aim to build a robust classifier that can effectively identify misleading or fabricated news.

## Clear goal(s)
1. Develop a binary classification model to predict whether a given news article is real or fake.
2. Compare BERT and DistilBERT to evaluate their effectiveness in fake news detection.
3. Implement a visualization dashboard to illustrate key insights from the dataset, such as the most common misleading words or sources.

## Data Collection & Source
- I will use the Fake News Prediction Dataset from **Kaggle**, which contains labeled news articles. Dataset link:*https://www.kaggle.com/datasets/rajatkumar30/fake-news*
### Dataset details:
- Columns: Title, text content, label (1 = Fake, 0 = Real)
- Size: 10,000 news articles

## Model Selection & Training Approach
I will experiment with two deep learning-based NLP models:
1. BERT (Bidirectional Encoder Representations from Transformers)
2. DistilBERT (A lightweight version of BERT)

## Data Visualization
- Word Cloud: Highlighting the most frequently used words in fake vs. real news.
- t-SNE Embedding Plot: Visualizing how fake and real news are distributed in a reduced dimensional space.

## Test plan
1. Train on 80% of the dataset, test on 20%.
2. Use Hugging Face Transformers for model implementation.
3. Fine-tune BERT/DistilBERT on the Kaggle dataset.
### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC Curve to assess classification quality