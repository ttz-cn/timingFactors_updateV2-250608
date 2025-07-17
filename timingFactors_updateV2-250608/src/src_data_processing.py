import pandas as pd
import numpy as np
import datetime


# 获取start_date的前一个工作日
def get_previous_workday(date, Bday=1):
    start_date = pd.Timestamp(date)
    start_date_ = start_date - pd.offsets.BusinessDay(Bday)
    return datetime.strftime(start_date_, "%Y-%m-%d")


# 获取start_date所在月的上两个月月末
def get_previous_month_end(date, ME=2):
    start_date = pd.Timestamp(date)
    start_date_ = start_date - pd.offsets.MonthEnd(ME)
    return datetime.strftime(start_date_, "%Y-%m-%d")


# drop duplicates in raw data
def get_clean_raw_data(df):
    df.index = pd.to_datetime(df.index)
    return df[~df.index.duplicated(keep="first")]  # keep=first,保留首个行情数据
