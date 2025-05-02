from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Run a dummy inference
inputs = tokenizer("This is a fake news example.", return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)

# Check output shape (should be [1, 2] for binary classification)
assert outputs.logits.shape == (1, 2)
print("Test passed: output shape is", outputs.logits.shape)
