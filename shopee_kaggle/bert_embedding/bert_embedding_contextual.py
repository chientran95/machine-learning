import torch
from transformers import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
# Add the special tokens.
marked_text = "[CLS] " + text + " [SEP]"

import pdb
pdb.set_trace()

# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)
# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [1] * len(tokenized_text)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained(
    'bert-base-uncased',
    output_hidden_states = True, # Whether the model returns all hidden-states.
)
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]

token_embeddings = torch.stack(hidden_states, dim=0)
token_embeddings = torch.squeeze(token_embeddings, dim=1)
token_embeddings = token_embeddings.permute(1,0,2)
print(token_embeddings.size())

token_vecs_sum = []
for token in token_embeddings:
    sum_vec = torch.sum(token[-4:], dim=0)
    token_vecs_sum.append(sum_vec)
print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

print('First 5 vector values for each instance of "bank".')
print("bank vault   ", str(token_vecs_sum[6][:5]))
print("bank robber  ", str(token_vecs_sum[10][:5]))
print("river bank   ", str(token_vecs_sum[19][:5]))

from scipy.spatial.distance import cosine

# Calculate the cosine similarity between the word bank
# in "bank vault" vs "river bank" (different meaning).
vault_river = 1 - cosine(token_vecs_sum[6], token_vecs_sum[19])

# Calculate the cosine similarity between the word bank 
# in "bank robber" vs "river bank" (different meanings).
robber_river = 1 - cosine(token_vecs_sum[10], token_vecs_sum[19])

# Calculate the cosine similarity between the word bank
# in "bank robber" vs "bank vault" (same meaning).
robber_vault = 1 - cosine(token_vecs_sum[10], token_vecs_sum[6])

print('Vector similarity for *bank vault* and *river bank*:  %.2f' % vault_river)
print('Vector similarity for *bank robber* and *river bank*:  %.2f' % robber_river)
print('Vector similarity for *bank robber* and *bank vault*:  %.2f' % robber_vault)
