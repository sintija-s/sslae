import numpy as np
from torch import tensor
from torch.utils.data import DataLoader, Dataset


def twonn_dimension(data, return_xy=False):
    data = np.array(data)
    N = len(data)
    # mu = r2/r1 for each data point
    mu = []
    for i, x in enumerate(data):
        dist = np.sort(np.sqrt(np.sum((x - data) ** 2, axis=1)))
        r1, r2 = dist[dist > 0][:2]
        mu.append((i + 1, r2 / r1))

    # permutation function
    sigma_i = dict(
        zip(
            range(1, len(mu) + 1),
            np.array(sorted(mu, key=lambda x: x[1]))[:, 0].astype(int),
        )
    )

    mu = dict(mu)

    # cdf F(mu_{sigma(i)})
    F_i = {}
    for i in mu:
        F_i[sigma_i[i]] = i / N

    # fitting coordinates
    x = np.log([mu[i] for i in sorted(mu.keys())])
    y = np.array([1 - F_i[i] for i in sorted(mu.keys())])

    # avoid having log(0)
    x = x[y > 0]
    y = y[y > 0]

    y = -1 * np.log(y)

    # fit line through origin to get the dimension
    d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]

    if return_xy:
        return d, x, y
    else:
        return d


class TorchSSAEDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = tensor(X).float()
        self.Y = tensor(y).float()

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self) -> int:
        return len(self.X)


def construct_dataloaders(x, y, batch_size):
    """Create a dataloader from input data and labels.

    This follows the pattern in the original codebase."""
    dataset = TorchSSAEDataset(x, y)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    return dataloader


def generate_semi_supervised_batch(
    x_train_lab, x_train_unl, y_train_lab, y_train_unl, batch_size, unlabeled_proportion
):
    unlabeled_batchsize = int(np.ceil(batch_size * unlabeled_proportion))
    labeled_batchsize = batch_size - unlabeled_batchsize
    # ensure that there is at least one example inside
    if labeled_batchsize == 0:
        labeled_batchsize = 1
        unlabeled_batchsize -= 1
    x_train = []
    y_train = []

    labeled_data = list(zip(x_train_lab, y_train_lab))
    unlabeled_data = list(zip(x_train_unl, y_train_unl))

    while labeled_data and unlabeled_data:
        cnt_labeled = labeled_batchsize
        cnt_unlabeled = unlabeled_batchsize
        for _ in range(min(cnt_labeled, len(labeled_data))):
            x, y = labeled_data.pop()
            x_train.append(x)
            y_train.append(y)
            cnt_labeled -= 1

        for _ in range(min(cnt_unlabeled, len(unlabeled_data))):
            x, y = unlabeled_data.pop()
            x_train.append(x)
            y_train.append(y)
            cnt_unlabeled -= 1

    # Exhaust remaining points in both datasets, if any
    for _ in range(len(labeled_data)):
        x, y = labeled_data.pop()
        x_train.append(x)
        y_train.append(y)

    for _ in range(len(unlabeled_data)):
        x, y = unlabeled_data.pop()
        x_train.append(x)
        y_train.append(y)

    return np.array(x_train), np.array(y_train)
