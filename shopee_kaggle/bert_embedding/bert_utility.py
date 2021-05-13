import torch
from transformers import BertTokenizer, BertModel


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_sentence(sentence:str, max_length:int) -> (torch.Tensor, torch.Tensor):
    """Tokenize sentence using Bert Tokenizer

    Args:
        sentence (str): Sentence to tokenize
        max_length (int): Max length of sentences

    Returns:
        torch.Tensor, torch.Tensor: Tensor of tokens for sentence and Tensor of attention mask
    """
    encoded_dict = tokenizer.encode_plus(
        sentence,                      # Sentence to encode.
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        max_length = max_length,           # Pad & truncate all sentences.
        truncation=True,
        pad_to_max_length = True,
        return_attention_mask = True,   # Construct attn. masks.
        return_tensors = 'pt',     # Return pytorch tensors.
    )
    indexed_tokens = encoded_dict['input_ids']
    segments_ids = encoded_dict['attention_mask']
    return indexed_tokens, segments_ids
    
    # tokens_tensor = torch.cat((indexed_tokens, indexed_tokens, indexed_tokens), dim=0)
    # segments_tensors = torch.cat((segments_ids, segments_ids, segments_ids), dim=0)

def get_bert_embedding(tokens_tensor, segments_tensors, postprocess_op='avg'):
    """Get BERT embedding of words or sentence

    Args:
        tokens_tensor (torch.Tensor): Tensor of tokenized sentences with shape (<batch_size>, <sentence_length>)
        segments_tensors (torch.Tensor): Tensor of attention masks with shape (<batch_size>, <sentence_length>)
        postprocess_op (str, optional): Operation used to aggregate embedding ('avg' for sentence, 'concat' or 'sum' for words). Defaults to 'avg'.

    Returns:
        Tensor or [Tensor]: Embedding tensor for sentence or list of word embedding tensors
    """
    model = BertModel.from_pretrained(
        'bert-base-uncased',
        output_hidden_states = True, # Whether the model returns all hidden-states.
    )
    model.eval()
    model.cuda()

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0)
    # token_embeddings = torch.squeeze(token_embeddings, dim=1) # For single sentence
    token_embeddings = token_embeddings.permute(1,2,0,3)
    # `token_embeddings` is a [<sentences>, <text_len> x 13 x 768] tensor
    
    if postprocess_op == 'concat':
        embs = []
        for sentence in token_embeddings:
            # `sentence` is a [<text_len> x 13 x 768] tensor
            token_vecs_cat = []
            for token in sentence:
                # `token` is a [13 x 768] tensor (not count first input layer)
                # Concatenate the vectors (that is, append them together) from the last four layers.
                # Each layer vector is 768 values, so cat_vec (3072)
                cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

                # Use `cat_vec` to represent `token`.
                # token_vecs_cat (<text_len>, 3072)
                token_vecs_cat.append(cat_vec)
            # embs (<sentences>, <text_len>, 3072)
            embs.append(torch.stack(token_vecs_cat))
        return torch.stack(embs)
    elif postprocess_op == 'sum':
        embs = []
        for sentence in token_embeddings:
            token_vecs_sum = []
            for token in sentence:
                # sum_vec (768)
                sum_vec = torch.sum(token[-4:], dim=0)

                # Use `sum_vec` to represent `token`.
                # token_vecs_sum (<text_len>, 768)
                token_vecs_sum.append(sum_vec)
            # embs (<sentences>, <text_len>, 768)
            embs.append(torch.stack(token_vecs_sum))
        return torch.stack(embs)
    elif postprocess_op == 'avg':
        embs = []
        for sentence in token_embeddings:
            # `sentence` is a [<text_len> x 13 x 768] tensor
            # token_vecs (<text_len>, 768)
            token_vecs = sentence[:, -2, :]
            # sentence_embedding (768)
            sentence_embedding = torch.mean(token_vecs, dim=0)
            # embs (<sentences>, 768)
            embs.append(sentence_embedding)
        return torch.stack(embs)
