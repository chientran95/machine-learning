import os
import time
import shutil
import tensorflow as tf
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import DepthEstimationModel
from data_generator import DataGenerator


tf.random.set_seed(123)

annotation_folder = "/datasets/"
if not os.path.exists(os.path.abspath(".") + annotation_folder):
    os.makedirs(os.path.abspath(".") + annotation_folder)
    annotation_zip = tf.keras.utils.get_file(
        "val.tar.gz",
        cache_subdir=os.path.abspath("."),
        origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
        extract=True,
    )
    shutil.move('./val', os.path.abspath(".") + annotation_folder + 'val')

path = os.path.abspath(".") + annotation_folder + 'val/indoors'

filelist = []

for root, dirs, files in os.walk(path):
    for file in files:
        filelist.append(os.path.join(root, file))

filelist.sort()
data = {
    "image": [x for x in filelist if x.endswith(".png")],
    "depth": [x for x in filelist if x.endswith("_depth.npy")],
    "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
}
df = pd.DataFrame(data)

df = df.sample(frac=1, random_state=42)

HEIGHT = 256
WIDTH = 256
LR = 0.0002
EPOCHS = 100
BATCH_SIZE = 32

def visualize_depth_map(samples, test=False, model=None):
    input, target = samples
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    if test:
        pred = model.predict(input)
        fig, ax = plt.subplots(6, 3, figsize=(50, 50))
        for i in range(6):
            ax[i, 0].imshow((input[i].squeeze()))
            ax[i, 1].imshow((target[i].squeeze()), cmap=cmap)
            ax[i, 2].imshow((pred[i].squeeze()), cmap=cmap)

    else:
        fig, ax = plt.subplots(6, 2, figsize=(50, 50))
        for i in range(6):
            ax[i, 0].imshow((input[i].squeeze()))
            ax[i, 1].imshow((target[i].squeeze()), cmap=cmap)


visualize_samples = next(
    iter(DataGenerator(data=df, batch_size=6, dim=(HEIGHT, WIDTH)))
)
def sample_vis():
    visualize_depth_map(visualize_samples)

def pointcloud_vis():
    depth_vis = np.flipud(visualize_samples[1][1].squeeze())  # target
    img_vis = np.flipud(visualize_samples[0][1].squeeze())  # input

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection="3d")

    STEP = 3
    for x in range(0, img_vis.shape[0], STEP):
        for y in range(0, img_vis.shape[1], STEP):
            ax.scatter(
                [depth_vis[x, y]] * 3,
                [y] * 3,
                [x] * 3,
                c=tuple(img_vis[x, y, :3] / 255),
                s=3,
            )
        ax.view_init(45, 135)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LR,
    amsgrad=False,
)
model = DepthEstimationModel()
# Define the loss function
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
# Compile the model
model.compile(optimizer, loss=cross_entropy)

train_loader = DataGenerator(
    data=df[:260].reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
)
validation_loader = DataGenerator(
    data=df[260:].reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
)
model.fit(
    train_loader,
    epochs=EPOCHS,
    validation_data=validation_loader,
)
ts = time.time()
model.save('checkpoint/model_{}'.format(ts))

# test_loader = next(
#     iter(DataGenerator(data=df[265:].reset_index(drop="true"),
#                        batch_size=6, dim=(HEIGHT, WIDTH)))
# )
# visualize_depth_map(test_loader, test=True, model=model)

# test_loader = next(
#     iter(DataGenerator(data=df[300:].reset_index(drop="true"),
#                        batch_size=6, dim=(HEIGHT, WIDTH)))
# )
# visualize_depth_map(test_loader, test=True, model=model)
