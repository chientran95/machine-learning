import torch
from transformers import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

################################ Manual #################################
text = "After stealing money from the bank vault, the bank robber was seen " \
       "fishing on the Mississippi river bank."
# Add the special tokens.
marked_text = "[CLS] " + text + " [SEP]"

# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)
# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [1] * len(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
#########################################################################

################################ Function #################################
encoded_dict = tokenizer.encode_plus(
    text,                      # Sentence to encode.
    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
    max_length = 32,           # Pad & truncate all sentences.
    truncation=True,
    pad_to_max_length = True,
    return_attention_mask = True,   # Construct attn. masks.
    return_tensors = 'pt',     # Return pytorch tensors.
)
indexed_tokens = encoded_dict['input_ids']
segments_ids = encoded_dict['attention_mask']
tokens_tensor = indexed_tokens
segments_tensors = segments_ids
#########################################################################

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

token_vecs_cat = []
# `token_embeddings` is a [22 x 12 x 768] tensor (not count first input layer)
for token in token_embeddings:
    # `token` is a [12 x 768] tensor (not count first input layer)
    # Concatenate the vectors (that is, append them together) from the last four layers.
    # Each layer vector is 768 values, so `cat_vec` is length 3,072.
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    
    # Use `cat_vec` to represent `token`.
    token_vecs_cat.append(cat_vec)
print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))

token_vecs_sum = []
for token in token_embeddings:
    # Sum the vectors from the last four layers.
    sum_vec = torch.sum(token[-4:], dim=0)
    
    # Use `sum_vec` to represent `token`.
    token_vecs_sum.append(sum_vec)
print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))



token_vecs = token_embeddings[:, -2, :]
# Calculate the average of all 22 token vectors.
sentence_embedding = torch.mean(token_vecs, dim=0)
print ("Our final sentence embedding vector of shape:", sentence_embedding.size())
