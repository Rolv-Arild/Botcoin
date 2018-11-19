import bisect
import datetime
import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from example.simple_bitcoin_predictor import SimpleBitcoinPredictor


def normalize_data(data, from_zero=True):
    max_vals = list(data[0])
    if not from_zero:
        min_vals = list(data[0])
        for d in data[1:]:
            for i in range(len(d)):
                if d[i] > max_vals[i]:
                    max_vals[i] = d[i]
                if d[i] < min_vals[i]:
                    min_vals[i] = d[i]

        for d in data:
            for i in range(len(d)):
                d[i] = (d[i] - min_vals[i]) / (max_vals[i] - min_vals[i])

        return data, max_vals, min_vals
    else:
        for d in data[1:]:
            for i in range(len(d)):
                if d[i] > max_vals[i]:
                    max_vals[i] = d[i]

        for d in data:
            for i in range(len(d)):
                d[i] = d[i] / max_vals[i]

        return data, max_vals


def find_increase(data, index):
    return [[data[i + 1][index] / data[i][index] - 1] for i in range(len(data) - 1)]


def convert_timestamp(timestamp):
    date = datetime.datetime.utcfromtimestamp(timestamp)
    time = date.time()

    return date.year, date.month, date.day, date.weekday(), (time.hour * 60 + time.minute)


def generate_classes(y, k):
    y_sorted = sorted(y)
    cutoffs = [y_sorted[i * len(y_sorted) // k] for i in range(k)]
    cutoffs.append(y_sorted[-1])
    res = []
    for i in range(len(y)):
        res.append([0.0] * k)
        index = bisect.bisect_left(cutoffs, y[i]) - 1
        res[i][index] = 1.0
    return res


def get_full_data(k, interval):
    data = pandas.read_csv("../resources/2017-present.csv", dtype='float64')

    x = data.drop("Timestamp", 1)
    # x = data.filter(["Trend", "High_USD", "Change_USD", "Weighted_Price"], axis=1)
    # x = data.filter(["Weighted_Price"], axis=1)

    # x = data.get_values().tolist()
    # for r in x:
    #     r.pop(0)

    # Normalize data
    maxes = x.max()
    mins = x.min()
    xn = (x - mins) / (maxes - mins)

    x = x.values.tolist()
    x = x[::interval]

    y = find_increase(x, 6)
    y = generate_classes(y, k)
    # x = y[:-1]
    # y = x[1:]

    # xn = xn.drop("Weighted_Price", axis=1)
    x = xn.values.tolist()
    x = x[::interval]
    return np.array(x), np.array(y)


def get_data(k, interval):
    data = pandas.read_csv("../resources/2017-present.csv", dtype='float64')

    # x = data.drop("Timestamp", 1)
    x = data.filter(["Close"], axis=1)

    # x = data.get_values().tolist()
    # for r in x:
    #     r.pop(0)

    # Normalize data
    # maxes = x.max()
    # mins = x.min()
    # # xn = (x - mins) / (maxes - mins)

    x = x.values.tolist()
    x = x[::interval]

    y = find_increase(x, 0)
    y = generate_classes(y, k)
    x = y[:-1]
    y = x[1:]

    # x = xn.values.tolist()
    # x = x[::60]
    return np.array(x), np.array(y), data


def plot_prediction(session, model: SimpleBitcoinPredictor, data, x, y):
    global p
    preds = np.array([])
    for i in range(1, len(data)):
        p = session.run(model.f, {model.batch_size: 1, model.x: [data[:i]]})[0]
        np.append(preds, np.argmax(p))

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, ax = plt.subplots()

    cmap = ListedColormap([[1 - n / len(p), n / len(p), 0, 1] for n in range(len(p))])
    norm = BoundaryNorm(range(len(p)), cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(y)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    plt.show()
