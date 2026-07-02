import argparse
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import ClassVar

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
from settings import DATA_DIR


@dataclass(frozen=True)
class TrainingConfig:
    supported_datasets: ClassVar[tuple[str, ...]] = ("rmnist",)
    supported_models: ClassVar[tuple[str, ...]] = ("resnet", "lenet")
    experiment_root: ClassVar[Path] = DATA_DIR / "experiments"

    dataset: str = "rmnist"
    model: str = "resnet"
    seed: int = 31
    lr: float = 3e-3
    num_ep: int = 301
    gan_ratio: float = 0
    data_augmentation: bool = True
    experiment_dir: Path | None = None

    @classmethod
    def from_args(cls, args):
        return cls(
            dataset=args.dataset,
            model=args.model,
            seed=args.seed,
            lr=args.lr,
            num_ep=args.num_ep,
            gan_ratio=args.gan_ratio,
            data_augmentation=args.data_augmentation,
        )

    def with_experiment_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        augmentation = "aug" if self.data_augmentation else "noaug"
        gan_ratio = str(self.gan_ratio).replace(".", "p")
        base_name = (
            f"{timestamp}-{self.dataset}-{self.model}-{augmentation}"
            f"-gan{gan_ratio}-seed{self.seed}"
        )
        experiment_dir = self._unique_experiment_dir(base_name)
        return replace(self, experiment_dir=experiment_dir)

    @classmethod
    def _unique_experiment_dir(cls, base_name):
        experiment_dir = cls.experiment_root / base_name
        suffix = 1
        while experiment_dir.exists():
            experiment_dir = cls.experiment_root / f"{base_name}-{suffix}"
            suffix += 1
        return experiment_dir

    def export(self):
        if self.experiment_dir is None:
            raise ValueError("experiment_dir must be set before exporting config")

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.experiment_dir / "training_config.json"
        config_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return config_path

    def to_dict(self):
        if self.experiment_dir is None:
            experiment_dir = None
        else:
            experiment_dir = str(self.experiment_dir.relative_to(DATA_DIR))

        return {
            "dataset": self.dataset,
            "model": self.model,
            "seed": self.seed,
            "lr": self.lr,
            "num_ep": self.num_ep,
            "gan_ratio": self.gan_ratio,
            "data_augmentation": self.data_augmentation,
            "experiment_dir": experiment_dir,
        }


def build_parser():
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train the MNIST one-shot model")
    parser.add_argument(
        "--dataset", choices=TrainingConfig.supported_datasets, default=defaults.dataset
    )
    parser.add_argument(
        "--model", choices=TrainingConfig.supported_models, default=defaults.model
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--num_ep", type=int, default=defaults.num_ep)
    parser.add_argument("--gan_ratio", type=float, default=defaults.gan_ratio)
    parser.add_argument(
        "--data-augmentation",
        action=argparse.BooleanOptionalAction,
        default=defaults.data_augmentation,
    )
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


def deep_learning(dataset, config):
    dataset = preprocess_dataset(
        dataset,
        seed=config.seed,
        gan_ratio=config.gan_ratio,
        data_augmentation=config.data_augmentation,
        output_dir=config.experiment_dir,
    )
    model, image_size = make_model(config.model)
    fit_and_evaluate(
        dataset,
        model,
        lr=config.lr,
        num_ep=config.num_ep,
        image_size=image_size,
        output_dir=config.experiment_dir,
        model_name=config.model,
    )


def siamese_net(dataset, config):
    """buggy"""

    def preprocess(dataset, data_augmentation=True):
        train_x, train_y, test_x, test_y = dataset
        train_x = train_x.reshape((-1, 28, 28, 1))
        test_x = test_x.reshape((-1, 28, 28, 1))

        if data_augmentation:
            n_samples = 1024
            # data augmentation with Augmentor
            p = Augmentor.Pipeline()
            p.set_seed(config.seed)
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
        optimizer = optim.Adam(net.parameters(), lr=config.lr)
        batch = 128
        running_loss = 0
        for epoch in range(config.num_ep):
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
    if parsed_args is None:
        parsed_args = build_parser().parse_args()
    config = TrainingConfig.from_args(parsed_args).with_experiment_dir()
    config_path = config.export()
    print(f"Experiment directory: {config.experiment_dir.relative_to(DATA_DIR)}")
    print(f"Training config: {config_path.relative_to(DATA_DIR)}")

    # seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    # dataset
    dataset = make_train_dataset(config.dataset)
    # statistical_ml(dataset)
    deep_learning(dataset, config)
    # siamese_net(dataset, config)


if __name__ == "__main__":
    main()
