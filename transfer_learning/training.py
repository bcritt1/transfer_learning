import os
import torch
from transformers import BertTokenizer, BertModel

# Load the corpus
user = os.getenv('USER')
corpusdir = '/scratch/users/{}/corpus/'.format(user)
corpus = []
for infile in os.listdir(corpusdir):
    with open(corpusdir + infile, errors='ignore') as fin:
        corpus.append(fin.read())

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Tokenize the corpus
tokenized_corpus = tokenizer(corpus, padding="max_length", truncation=True, return_tensors="pt")

# Load the pre-trained BERT model
model = BertModel.from_pretrained("bert-base-cased")

# Set the model to evaluation mode
model.eval()

# Forward pass to get the encoded representations
with torch.no_grad():
    inputs = tokenized_corpus.input_ids
    attention_mask = tokenized_corpus.attention_mask
    encoded_layers, _ = model(inputs, attention_mask=attention_mask)

# Process the encoded representations as needed for your downstream task
# For example, you can use the encoded layers as input to another model or perform clustering, similarity calculations, etc.

# Save the encoded representations
output_dir = "/scratch/users/{}/models/".format(user)
os.makedirs(output_dir, exist_ok=True)
torch.save(encoded_layers, os.path.join(output_dir, "encoded_layers.pt"))