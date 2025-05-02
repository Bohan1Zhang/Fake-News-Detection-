# Final Report
## 10-minute presentation
Video link:*https://www.youtube.com/watch?v=mtCwXTV6LLY*
## How to Build and Run the Code
A Makefile (macOS/Linux) and Makefile.win (Windows) are provided for setup. They create a virtual environment named venv, install dependencies, and register a Jupyter kernel.
### macOS/Linux:
```
make install
make notebook
```
### Windows:
```
make -f Makefile.win install
make -f Makefile.win notebook
```
## Test Code and GitHub Workflow
A minimal test script test_model.py is included to verify that the model and tokenizer can be successfully loaded and used for inference. It does not require training and is meant to confirm that the environment and model pipeline are functional.

To run the test script after installation:
### macOS/Linux:
```
make test
```
### Windows:
```
make -f Makefile.win test

```
This runs a dummy input ("This is a fake news example.") through the DistilBERT model and checks that the output shape matches the expected [1, 2] for binary classification. The test will print a confirmation message if successful:Test passed: output shape is torch.Size([1, 2])
## Data Visualization

1. **Text Length Distribution**  
   Before tokenization, we plotted the number of tokens per article to guide `max_length` selection.  
   Most articles fall under 1500 tokens, with a strong peak around **500**.

2. **BERT Token Distribution**  
   After applying the `bert-base-uncased` tokenizer, we observed that the majority of samples still stay within **512 tokens**.  
   This confirmed our choice of `max_length = 512`.

3. **Word Cloud Comparison**  
   We generated word clouds for both fake and real news headlines:

   - Real news prominently featured terms like **said**, **reuters**, **president**, and **trump**.  
   - Fake news shared many top terms but also highlighted words such as **people**, **clinton**, and **obama** with different contextual emphasis.  

   This helped us understand lexical patterns relevant to classification.

4. **Confusion Matrix**  
   We visualized the model's performance using a confusion matrix.  
   The classifier shows high accuracy with **minimal misclassifications** on the test set.

5. **Loss Curve**  
   A combined loss plot tracked training and test loss over steps:

   - The model quickly converged within the **first epoch**.  
   - Loss remained **low and stable** in later epochs, with no signs of overfitting.

6. **Confidence Distribution**  
   We plotted the max softmax probability for each test prediction.  
   Most predictions were made with very **high confidence** (above **0.95**),  
   indicating the model is well-calibrated for this binary task.

7. **Robustness Check (Perturbation Test)**  
   When the **first sentence** of each test sample was removed,  
   performance dropped from ~**100%** to ~**77%**.  
   This suggests that early-sentence information plays a crucial role in model decision-making  
   and highlights the importance of **full input context** for consistent performance.


## Data Processing and Modeling

We followed a structured pipeline to prepare the dataset for training and evaluation.

1. **Merging Real and Fake News**  
   We combined `True.csv` and `Fake.csv` from Kaggle and assigned binary labels:  
   - `1` for real news  
   - `0` for fake news  
   The combined dataset was shuffled (`random_state=42`) to ensure reproducibility, then saved as `News.csv`.

2. **Token Length Analysis**  
   Before tokenization, we computed word counts per article to understand raw text lengths.  
   This analysis supported our decision to set `max_length = 512` for BERT inputs.

3. **BERT Token Analysis**  
   Using the `bert-base-uncased` tokenizer, we encoded texts without truncation or padding.  
   The resulting token length distribution showed that the majority of articles fit within **512 tokens**, confirming this limit as appropriate.

4. **Text Preprocessing for Word Cloud**  
   We cleaned and tokenized the text data using the following steps:  
   - Converted all text to lowercase  
   - Removed punctuation using `string.punctuation`  
   - Removed English stopwords via `nltk.corpus.stopwords`  
   - Tokenized the cleaned text  
   - Counted word frequencies using `collections.Counter`  
   This was done separately for real and fake news samples to allow comparative visualization.

5. **Word Cloud Visualization**  
   We visualized the top 50 most frequent words for both classes using `WordCloud`.  
   The resulting differences highlighted distinct lexical patterns that might inform model learning.

6. **Dataset Construction**  
   We created a custom PyTorch dataset `NewsDataset` with the following features:  
   - Tokenization using the HuggingFace tokenizer  
   - `truncation=True` and `max_length=512`  
   - `padding='max_length'` to ensure uniform input shapes  
   Each dataset entry includes:  
   - `input_ids`  
   - `attention_mask`  
   - `label`  

   We then performed an **80-20 train-test split** and created `train_dataset` and `test_dataset` accordingly.

## Results and Achievements

We successfully built and evaluated a DistilBERT-based classifier for fake news detection. Below are the outcomes and key observations:

1. **Classification Performance**  
   The model achieved nearly perfect results on the test set:  
   - **Accuracy**: 100% on the original test split  
   - **Confusion Matrix** showed only 1 misclassification out of 2000 samples  
   - Metrics:  
     - Precision: 1.00  
     - Recall: 1.00  
     - F1-score: 1.00  

2. **Robustness Evaluation**  
   To evaluate generalization, we ran two tests:  
   - Repeated training-test splits using different `random_state` values  
   - Perturbation test: removed the first sentence from each article  
   > Accuracy dropped from ~100% to **77.15%** under perturbation, highlighting reliance on initial sentence cues.

3. **Loss and Confidence Curves**  
   - Loss curves remained stable with no signs of overfitting  
   - Confidence distribution showed most predictions were made with **very high certainty** (softmax > 0.95)

4. **Data Insight Visualizations**  
   - Word clouds revealed **distinct linguistic patterns** in real vs. fake news  
   - BERT token count distribution justified our use of `max_length = 512`  

5. **Project Goal Alignment**  
   -  Built an accurate fake news classifier using DistilBERT  
   -  Visualized lexical patterns that may influence misclassification  
   -  Confirmed the model's robustness and areas for improvement via perturbation analysis  

The project demonstrates that even a lightweight model like DistilBERT, when fine-tuned properly, can achieve strong performance on real-world fake news detection tasks.



# Fake-News-Detection-
![Fake News](./png_file/news.png)
## Description of the project
The rapid spread of misinformation has become a pressing issue in the digital age, particularly with the rise of social media and online news platforms. Fake news can have serious consequences, from political manipulation to public misinformation about health crises. The goal of this project is to develop a machine learning model capable of distinguishing between real and fake news articles. By leveraging **BERT / DistilBERT**, a state-of-the-art NLP model, we aim to build a robust classifier that can effectively identify misleading or fabricated news.

## Clear goal(s)
1. Develop a binary classification model to predict whether a given news article is real or fake.
2. Use DistilBERT to evaluate their effectiveness in fake news detection.
3. Implement a visualization dashboard to illustrate key insights from the dataset, such as the most common misleading words or sources.

## Data Collection & Source
- I will use the Fake News Prediction Dataset from **Kaggle**, which contains labeled news articles. Dataset link:*https://www.kaggle.com/datasets/jainpooja/fake-news-detection?select=True.csv*
### Dataset details:
- Columns: Title, text content, label (1 = Fake, 0 = Real)
- Size: 44,898 news articles

## Model Selection & Training Approach
I will experiment with two deep learning-based NLP models:
1. DistilBERT (A lightweight version of BERT)

## Data Visualization
- Word Cloud: Highlighting the most frequently used words in fake vs. real news.


## Test plan
1. Train on 80% of the dataset, test on 20%.
2. Use Hugging Face Transformers for model implementation.
3. Fine-tune DistilBERT on the Kaggle dataset.
### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-score



