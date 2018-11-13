import bisect
import datetime
import pandas
import numpy as np


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


def get_full_data(k):
    data = pandas.read_csv("../resources/2017-present.csv", dtype='float64')

    x = data.drop("Timestamp", 1)
    # x = data.filter(["Weighted_Price"], axis=1)

    # x = data.get_values().tolist()
    # for r in x:
    #     r.pop(0)

    # Normalize data
    maxes = x.max()
    mins = x.min()
    xn = (x - mins) / (maxes - mins)

    x = x.values.tolist()
    x = x[::60]

    y = find_increase(x, 6)
    y = generate_classes(y, k)
    # x = y[:-1]
    # y = x[1:]

    x = xn.values.tolist()
    x = x[::60]
    return np.array(x), np.array(y)


def get_data(k):
    data = pandas.read_csv("../resources/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv", dtype='float64')

    # x = data.drop("Timestamp", 1)
    x = data.filter(["Weighted_Price"], axis=1)

    # x = data.get_values().tolist()
    # for r in x:
    #     r.pop(0)

    x = x.tail(len(x) - 2625376)  # start of 2017

    # Normalize data
    # maxes = x.max()
    # mins = x.min()
    # # xn = (x - mins) / (maxes - mins)

    x = x.values.tolist()
    x = x[::60]

    y = find_increase(x, -1)
    y = generate_classes(y, k)
    x = y[:-1]
    y = x[1:]

    # x = xn.values.tolist()
    # x = x[::60]
    return np.array(x), np.array(y)
