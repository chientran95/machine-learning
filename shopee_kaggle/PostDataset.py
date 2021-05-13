from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils
from bert_embedding.bert_utility import tokenize_sentence


class PostDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = pd.read_csv(csv_file)
        tmp = self.dataframe.groupby('label_group').posting_id.agg('unique').to_dict()
        self.dataframe['target'] = self.dataframe.label_group.map(tmp)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.dataframe.image.iloc[idx])
        image = Image.open(img_name)
        # post_id = self.dataframe.posting_id.iloc[idx]
        # image_phash = self.dataframe.image_phash.iloc[idx]
        title = self.dataframe.title.iloc[idx]
        # label_group = self.dataframe.label_group.iloc[idx]
        # target = self.dataframe.target.iloc[idx]
        token_tensor, attn_mask = tokenize_sentence(title, 80)

        sample = {
            'image': image,
            'token_tensor': token_tensor[0],
            'attn_mask': attn_mask[0],
        }

        if self.transform:
            transformed_image = self.transform(sample['image'])
            sample = {
                'image': transformed_image,
                'token_tensor': token_tensor[0],
                'attn_mask': attn_mask[0],
            }

        return sample