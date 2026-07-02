import Augmentor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

from settings import OUTPUT_DIR
from tools.gan_augment import gan_augment
from visualization import plot_samples


def augment_training_data(train_x, train_y, seed, n_samples=1024):
    p = Augmentor.Pipeline()
    p.set_seed(seed)
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
    p.random_distortion(probability=0.8, grid_width=3, grid_height=3, magnitude=2)
    p.skew(probability=0.8, magnitude=0.3)
    p.shear(probability=0.5, max_shear_left=3, max_shear_right=3)
    generator = p.keras_generator_from_array(train_x, train_y, n_samples, scaled=False)

    train_x, train_y = next(generator)
    train_x = np.clip(train_x, 0, 1)
    return train_x, train_y


def add_gan_samples(train_x, train_y, seed, gan_ratio):
    if gan_ratio <= 0:
        return train_x, train_y

    n_samples = int(gan_ratio * len(train_x))
    augment_x, augment_y = gan_augment(train_x, train_y, seed, n_samples)
    train_x = np.concatenate([train_x, augment_x])
    train_y = np.concatenate([train_y, augment_y])
    return train_x, train_y


def to_image_tensors(train_x, train_y, test_x, test_y):
    train_x = np.transpose(train_x, (0, 3, 1, 2))
    test_x = np.transpose(test_x, (0, 3, 1, 2))
    train_x = torch.Tensor(train_x)
    test_x = torch.Tensor(test_x)
    train_y = torch.LongTensor(train_y)
    test_y = torch.LongTensor(test_y)
    return train_x, train_y, test_x, test_y


def preprocess_dataset(dataset, seed, gan_ratio=0, data_augmentation=True):
    train_x, train_y, test_x, test_y = dataset
    train_x = train_x.reshape((-1, 28, 28, 1))
    test_x = test_x.reshape((-1, 28, 28, 1))
    plot_samples(train_x, OUTPUT_DIR / f"origin-{seed}.png")

    if data_augmentation:
        train_x, train_y = augment_training_data(train_x, train_y, seed)
        plot_samples(train_x[:50], OUTPUT_DIR / "data_augmentation.png")

    train_x, train_y = add_gan_samples(train_x, train_y, seed, gan_ratio)
    return to_image_tensors(train_x, train_y, test_x, test_y)


def fit_and_evaluate(dataset, net, lr, num_ep, image_size=224):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    train_x, train_y, test_x, test_y = dataset

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)
    batch = 64
    running_loss = 0
    for epoch in range(num_ep):
        for i in range(len(train_x) // batch):
            idx = np.random.choice(range(len(train_x)), batch)
            x = train_x[idx].to(device)
            if image_size is not None:
                x = F.interpolate(x, size=image_size)
            y = train_y[idx].to(device)
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            running_loss = 0.99 * running_loss + 0.01 * train_loss
        if epoch % 20 == 0:
            print(running_loss)

    batch = 100
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_x), batch):
            x = test_x[i : i + batch].to(device)
            if image_size is not None:
                x = F.interpolate(x, size=image_size)
            out = net(x)
            preds.extend(out.argmax(1).cpu().numpy().tolist())
    print("accuracy: %.4f" % accuracy_score(test_y.cpu().numpy().tolist(), preds))
