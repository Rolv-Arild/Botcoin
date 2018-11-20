from util import get_data_average

df = get_data_average(60)
df.to_csv('../resources/2017-present-1hour.csv', index=False)
df = get_data_average(24 * 60)
df.to_csv('../resources/2017-present-1day.csv', index=False)
