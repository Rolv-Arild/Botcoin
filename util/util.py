import datetime


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
    return [[data[i][index] / data[i - 1][index] - 1] for i in range(1, len(data))]


def convert_timestamp(timestamp):
    date = datetime.datetime.utcfromtimestamp(timestamp)
    time = date.time()

    return date.year, date.month, date.day, date.weekday(), (time.hour * 60 + time.minute)
