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
    global data
    if interval == 1:
        data = pandas.read_csv("../resources/2017-present.csv", dtype='float64')
        data = data.drop("Timestamp", 1)
    elif interval == 60:
        data = pandas.read_csv("../resources/2017-present-1hour.csv", dtype='float64')
    elif interval == 24 * 60:
        data = pandas.read_csv("../resources/2017-present-1day.csv", dtype='float64')
    else:
        data = get_data_average(interval)

    x = data.pct_change().fillna(0).replace([float('Inf'), -float('Inf')], 0)

    # Normalize data
    # maxes = x.max()
    # mins = x.min()
    # x = (x - mins) / (maxes - mins)

    # index = x.columns.get_loc("Close")
    # x = x.values.tolist()

    y = x.filter(["Close"], axis=1).values.tolist()[2:]
    x = x.values.tolist()[1:-1]
    # y = find_increase(x, index)
    y = generate_classes(y, k)

    # x = x.values.tolist()

    return np.array(x), np.array(y), data


def get_data_average(interval):
    data = pandas.read_csv("../resources/2017-present.csv", dtype='float64')
    x = data.drop("Timestamp", 1)
    x = x.drop("Weighted_Price", 1)
    # x = x.values.tolist()
    newlist = pandas.DataFrame(columns=x.columns)
    for r in range(0, len(x) - interval, interval):
        open_btc = x.loc[r]["Open"]
        open_btc_usdt = x.loc[r]["Open_BTC"]
        open_eth = x.loc[r]["Open_ETH"]
        open_ltc = x.loc[r]["Open_LTC"]
        open_xrp = x.loc[r]["Open_XRP"]
        open_dji = x.loc[r]["Open_DJI"]
        open_gold = x.loc[r]["Open_gold"]
        open_usd = x.loc[r]["Open_USD"]

        high_btc = x.loc[r:interval + r]["High"].max()
        high_btc_usdt = x.loc[r:interval + r]["High_BTC"].max()
        high_eth = x.loc[r:interval + r]["High_ETH"].max()
        high_ltc = x.loc[r:interval + r]["High_LTC"].max()
        high_xrp = x.loc[r:interval + r]["High_XRP"].max()
        high_dji = x.loc[r:interval + r]["High_DJI"].max()
        high_gold = x.loc[r:interval + r]["High_gold"].max()
        high_usd = x.loc[r:interval + r]["High_USD"].max()

        low_btc = x.loc[r:interval + r]["Low"].min()
        low_btc_usdt = x.loc[r:interval + r]["Low_BTC"].min()
        low_eth = x.loc[r:interval + r]["Low_ETH"].min()
        low_ltc = x.loc[r:interval + r]["Low_LTC"].min()
        low_xrp = x.loc[r:interval + r]["Low_XRP"].min()
        low_dji = x.loc[r:interval + r]["Low_DJI"].min()
        low_gold = x.loc[r:interval + r]["Low_gold"].min()
        low_usd = x.loc[r:interval + r]["Low_USD"].min()

        close_btc = x.loc[interval + r - 1]["Close"]
        close_btc_usdt = x.loc[interval + r - 1]["Close_BTC"]
        close_eth = x.loc[interval + r - 1]["Close_ETH"]
        close_ltc = x.loc[interval + r - 1]["Close_LTC"]
        close_xrp = x.loc[interval + r - 1]["Close_XRP"]
        close_dji = x.loc[interval + r - 1]["Close_DJI"]

        volume_btc = x.loc[r:interval + r]["Volume_.BTC."].sum()
        volume_btc_cur = x.loc[r:interval + r]["Volume_.Currency."].sum()
        volume_btc_usdt = x.loc[r:interval + r]["Volume_BTC"].sum()
        volume_eth = x.loc[r:interval + r]["Volume_ETH"].sum()
        volume_ltc = x.loc[r:interval + r]["Volume_LTC"].sum()
        volume_xrp = x.loc[r:interval + r]["Volume_XRP"].sum()
        volume_dji = x.loc[r]["Volume_DJI"].sum()
        volume_usd = x.loc[r]["Volume_USD"].sum()

        trend = x.loc[r]["Trend"]
        change_usd = x.loc[r]["Change_USD"]
        price_usd = x.loc[r]["Price_USD"]
        price_gold = x.loc[r]["Price_gold"]

        newRow = pandas.DataFrame(
            {"Open": [open_btc], "Open_BTC": [open_btc_usdt], "Open_ETH": [open_eth], "Open_LTC": [open_ltc],
             "Open_XRP": [open_xrp], "Open_DJI": [open_dji], "Open_gold": [open_gold], "Open_USD": [open_usd],
             "High": [high_btc], "High_BTC": [high_btc_usdt], "High_ETH": [high_eth], "High_LTC": [high_ltc],
             "High_XRP": [high_xrp], "High_DJI": [high_dji], "High_gold": [high_gold], "High_USD": [high_usd],
             "Low": [low_btc], "Low_BTC": [low_btc_usdt], "Low_ETH": [low_eth], "Low_LTC": [low_ltc],
             "Low_XRP": [low_xrp], "Low_DJI": [low_dji], "Low_gold": [low_gold], "Low_USD": [low_usd],
             "Close": [close_btc], "Close_BTC": [close_btc_usdt], "Close_ETH": [close_eth], "Close_LTC": [close_ltc],
             "Close_XRP": [close_xrp], "Close_DJI": [close_dji], "Volume_.BTC.": [volume_btc],
             "Volume_.Currency.": [volume_btc_cur], "Volume_BTC": [volume_btc_usdt], "Volume_ETH": [volume_eth],
             "Volume_LTC": [volume_ltc], "Volume_XRP": [volume_xrp], "Volume_DJI": [volume_dji],
             "Volume_USD": [volume_usd], "Trend": [trend], "Change_USD": [change_usd], "Price_USD": [price_usd],
             "Price_gold": [price_gold]})
        newlist = pandas.concat([newlist, newRow], sort=True)

    return newlist


def get_data(k, interval):
    data = pandas.read_csv("../resources/2017-present.csv", dtype='float64')[::interval]

    # x = data.drop("Timestamp", 1)
    x = data.filter(["Close"], axis=1).pct_change().fillna(0).replace([float('Inf'), -float('Inf')], 0)

    # Normalize data
    # maxes = x.max()
    # mins = x.min()
    # x = (x - mins) / (maxes - mins)

    # index = x.columns.get_loc("Close")
    # x = x.values.tolist()

    y = x.filter(["Close"], axis=1).values.tolist()[2:]
    x = x.values.tolist()[1:-1]
    # y = find_increase(x, index)
    y = generate_classes(y, k)

    # x = x.values.tolist()

    return np.array(x), np.array(y), data


def get_reduced_data(k, interval):
    global data
    if interval == 1:
        data = pandas.read_csv("../resources/2017-present.csv", dtype='float64')
        data = data.drop("Timestamp", 1)
    elif interval == 60:
        data = pandas.read_csv("../resources/2017-present-1hour.csv", dtype='float64')
    elif interval == 24 * 60:
        data = pandas.read_csv("../resources/2017-present-1day.csv", dtype='float64')
    else:
        data = get_data_average(interval)

    cusd = data.filter(["Change_USD"], axis=1)
    x = data.filter(["Trend", "High_USD", "Close"], axis=1).pct_change()
    x = pandas.concat([x, cusd], sort=True).fillna(0).replace(
        [float('Inf'), -float('Inf')], 0)

    # Normalize data
    # maxes = x.max()
    # mins = x.min()
    # x = (x - mins) / (maxes - mins)

    # index = x.columns.get_loc("Close")
    # x = x.values.tolist()

    y = x.filter(["Close"], axis=1).values.tolist()[2:]
    x = x.drop("Close", 1).values.tolist()[1:-1]
    # y = find_increase(x, index)
    y = generate_classes(y, k)

    # x = x.values.tolist()

    return np.array(x), np.array(y), data


def plot_prediction(session, model, title, data, x, y):
    global p
    preds = []
    for i in range(len(data)):
        p = session.run(model.f, {model.batch_size: 1, model.x: [data[:i + 1]]})[0]
        preds.append(np.argmax(p))

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, ax = plt.subplots()

    cmap = ListedColormap(['r', 'y', 'g'])
    norm = BoundaryNorm([(len(p) - 1) * i / len(p) for i in range(len(p) + 1)], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.array(preds))
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Price")

    plt.show()
