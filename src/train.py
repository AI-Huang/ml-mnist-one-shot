import argparse

import Augmentor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from datasets.rmnist import make_dataset as make_rmnist_dataset
from models.nets import LeNet, MnistResNet, SiameseNet
from pipelines.deep_learning import fit_and_evaluate, preprocess_dataset

args = None


def build_parser():
    parser = argparse.ArgumentParser(description="Train the MNIST one-shot model")
    parser.add_argument("--dataset", choices=["rmnist"], default="rmnist")
    parser.add_argument("--model", choices=["resnet", "lenet"], default="resnet")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--num_ep", type=int, default=301)
    parser.add_argument("--gan_ratio", type=float, default=0)
    return parser


def make_train_dataset(name):
    if name == "rmnist":
        return make_rmnist_dataset()
    raise ValueError(f"Unsupported dataset: {name}")


def make_model(name):
    if name == "resnet":
        return MnistResNet(), 224
    if name == "lenet":
        return LeNet(), None
    raise ValueError(f"Unsupported model: {name}")


def statistical_ml(dataset):
    """classic ML algorithms"""

    def fit_and_evaluate(model):
        train_x, train_y, test_x, test_y = dataset
        model.fit(train_x, train_y)
        test_pred = model.predict(test_x)
        print("accuracy: %.4f" % accuracy_score(test_y, test_pred))

    # statistical ML models
    print("\nRandomForestClassifier")
    rf = RandomForestClassifier(n_estimators=200, max_features=10)
    fit_and_evaluate(rf)

    print("\nLogisticRegression")
    lr = LogisticRegression(solver="lbfgs", multi_class="auto")
    fit_and_evaluate(lr)

    print("\nGradientBoostingClassifier")
    gbdt = GradientBoostingClassifier()
    fit_and_evaluate(gbdt)

    print("\nSVMClassifier")
    svc = SVC()
    fit_and_evaluate(svc)

    print("\nKNeighborsClassifier")
    knn = KNeighborsClassifier(n_neighbors=1)
    fit_and_evaluate(knn)


def deep_learning(dataset):
    dataset = preprocess_dataset(
        dataset, seed=args.seed, gan_ratio=args.gan_ratio, data_augmentation=True
    )
    model, image_size = make_model(args.model)
    fit_and_evaluate(
        dataset, model, lr=args.lr, num_ep=args.num_ep, image_size=image_size
    )


def siamese_net(dataset):
    """buggy"""

    def preprocess(dataset, data_augmentation=True):
        train_x, train_y, test_x, test_y = dataset
        train_x = train_x.reshape((-1, 28, 28, 1))
        test_x = test_x.reshape((-1, 28, 28, 1))

        if data_augmentation:
            n_samples = 1024
            # data augmentation with Augmentor
            p = Augmentor.Pipeline()
            p.set_seed(args.seed)
            p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
            p.random_distortion(
                probability=0.8, grid_width=3, grid_height=3, magnitude=2
            )
            p.skew(probability=0.8, magnitude=0.3)
            p.shear(probability=0.5, max_shear_left=3, max_shear_right=3)
            generator = p.keras_generator_from_array(
                train_x, train_y, n_samples, scaled=False
            )

            origin_x, origin_y = train_x, train_y
            train_x, train_y = next(generator)
            train_x = np.clip(train_x, 0, 1)

        # convert to tensr
        train_x = torch.Tensor(train_x.reshape((-1, 784)))
        origin_x = torch.Tensor(origin_x.reshape((-1, 784)))
        test_x = torch.Tensor(test_x.reshape(-1, 784))
        train_y = torch.LongTensor(train_y)
        origin_y = torch.LongTensor(origin_y)
        test_y = torch.LongTensor(test_y)
        return train_x, train_y, test_x, test_y, origin_x, origin_y

    def fit_and_evaluate(dataset, net):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = net.to(device)

        train_x, train_y, test_x, test_y, origin_x, origin_y = dataset
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        batch = 128
        running_loss = 0
        for epoch in range(args.num_ep):
            for i in range(len(train_x) // batch):
                idx = np.random.choice(range(len(train_x)), batch * 2)
                x, y = train_x[idx].to(device), train_y[idx].to(device)
                x1, x2 = torch.split(x, batch, dim=0)
                y1, y2 = torch.split(y, batch, dim=0)
                y = (y1 == y2).float()

                optimizer.zero_grad()
                out = net(x1, x2)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
                running_loss = 0.99 * running_loss + 0.01 * train_loss
            if epoch % 20 == 0:
                print(running_loss)

        batch = 1000
        preds = []
        with torch.no_grad():
            x_, _ = origin_x.to(device), origin_y.to(device)
            for i in range(0, len(test_x), batch):
                x = test_x[i : i + batch].to(device)
                out = net.predict(x, x_)
                preds.extend(out.argmax(1).cpu().numpy().tolist())
        print("accuracy: %.4f" % accuracy_score(test_y.cpu().numpy().tolist(), preds))

    dataset = preprocess(dataset, data_augmentation=True)
    fit_and_evaluate(dataset, SiameseNet())


def main(parsed_args=None):
    global args
    if parsed_args is None:
        parsed_args = build_parser().parse_args()
    args = parsed_args

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # dataset
    dataset = make_train_dataset(args.dataset)
    # statistical_ml(dataset)
    deep_learning(dataset)
    # siamese_net(dataset)


if __name__ == "__main__":
    main()
