import numpy as np


def bagging(n_bags, pred_fn):
    train_bag_preds = []
    test_bag_preds = []

    for n in range(n_bags):
        print("Start Bagging %s" % n)
        train_preds, test_preds = pred_fn(n)
        train_bag_preds.append(train_preds)
        test_bag_preds.append(test_preds)

    train_bag_preds = np.array(train_bag_preds).T
    test_bag_preds = np.concatenate(test_bag_preds, axis=1)

    print("Train preds: ", train_bag_preds.shape)
    print("Test preds: ", test_bag_preds.shape)

    return train_bag_preds, test_bag_preds
