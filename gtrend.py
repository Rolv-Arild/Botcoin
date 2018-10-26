import pytrends
import pandas as pd
from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["Bitcoin"]

dataframe = pytrends.get_historical_interest(kw_list, year_start=2012, month_start=1, day_start=1, year_end=2014, month_end=12, day_end=30, cat=0, geo='US', gprop='', sleep=0)
pd.DataFrame(dataframe)

pd.DataFrame.to_csv(dataframe, 'bitcoin trend 2012-2014.csv')
