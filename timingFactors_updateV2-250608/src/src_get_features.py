import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import talib as ta
import datetime
from functools import reduce


def calculate_not_change(df, **kwargs):
    df["features"] = df["CLOSE"]
    return df


def calculate_pct_change(df, **kwargs):
    df["features"] = df["CLOSE"].pct_change(1)
    return df


#  Calculates a weekly Exponential Moving Average (EWM) from daily factor data
def calculate_day2week_ewm(df, **kwargs):

    freq = kwargs.get("freq", "B")
    span = kwargs.get("span", 5)
    if_intra_week = kwargs.get("if_intra_week", True)

    factor = df[["Date", "CLOSE"]].set_index("Date")  # Meta-Data中都是CLOSE
    factor.index = pd.to_datetime(factor.index)  # ini_index
    factor = factor.resample(freq).last().ffill()  # 预处理，补全工作日保持数据连续
    factor_ = factor.rolling(span).apply(
        lambda row: row.ewm(span=span, adjust=True).mean().iloc[-1]
    )  # 周内窗口指数平均
    """
    这里有两个resample，我来分别说一下是干什么的：
    第一个resample-->由于周五才能完整代表一整周，所以我们要用周五的ewm值当作本周值，只能用w-sun/fri.last（）；
    第二个resample-->上边操作后，由于signal数据是要对齐周一的行情数据的（操作周一的行情数据），所以需要再把周末频的ts转换为周一频率的ts；
    而w-resample中（m又不一样），是以标签结尾为一个采样周期的（即这周而二到下周一是一个采样周期），因此w-resample后，周日的数据会变到下周一，需要shift（-1）回来
    这里也可以用timedelta对index操作【factor_.index = factor_.index - pd.Timedelta(days=6) 】
    """
    factor_ = factor_.resample("W-FRI").last().resample("W-MON").last().shift(-1)  # 用周一作为index
    # factor_ = factor_[factor_.index < end_date].dropna()  # 日期截取
    # 如果需要周与周的移动平均
    if if_intra_week:
        intra_span = kwargs.get("intra_span", 8)
        factor_.ewm(span=intra_span, adjust=True).mean().ffill()  # ffill用来清理一周都是非工作日的情况
    factor_["features"] = factor.CLOSE  # 重新命名为features
    # factor_["label"] = df.label.iloc[0]  # 添加标签列
    factor_.reset_index(inplace=True)  # 重置index
    return factor_


def get_sma_idc(df, timeperiod=20, if_resample=True, **kwargs):
    column_name = kwargs.get("column_name", "CLOSE")  # 默认使用CLOSE列
    if if_resample:
        data_temp = (df[column_name].resample("B").last().ffill()).to_frame()  # 确保是DataFrame格式
    else:
        data_temp = df[column_name].to_frame()
    data_temp["sma"] = ta.SMA(data_temp[column_name].values, timeperiod)
    return data_temp.reset_index()


def get_bbi_idc(df, tp1=3, tp2=6, tp3=12, tp4=24, **kwargs):
    column_name = kwargs.get("column_name", "CLOSE")  # 默认使用CLOSE列
    data_temp = (df[column_name].resample("B").last().ffill()).to_frame()  # 确保是DataFrame格式
    bbi_n1 = ta.SMA(data_temp["CLOSE"].values, timeperiod=tp1)
    bbi_n2 = ta.SMA(data_temp["CLOSE"].values, timeperiod=tp2)
    bbi_n3 = ta.SMA(data_temp["CLOSE"].values, timeperiod=tp3)
    bbi_n4 = ta.SMA(data_temp["CLOSE"].values, timeperiod=tp4)
    data_temp["bbi"] = (bbi_n1 + bbi_n2 + bbi_n3 + bbi_n4) / 4
    return data_temp.reset_index()


def get_cci_idc(df, timeperiod=10, **kwargs):
    column_name = kwargs.get("column_name", "CLOSE")  # 默认使用CLOSE列
    data_temp = (df[column_name].resample("B").last().ffill()).to_frame()  # 确保是DataFrame格式
    data_temp["cci"] = ta.CCI(
        data_temp["CLOSE"].values, data_temp["CLOSE"].values, data_temp["CLOSE"].values, timeperiod
    )
    return data_temp.reset_index()


def get_kdj_idc(df, fastk_period=9, slowk_period=3, slowd_period=3, **kwargs):

    column_name = kwargs.get("column_name", ["HIGH", "CLOSE", "LOW"])  # 默认使用CLOSE列
    data_temp = df[column_name].resample("B").last().ffill()

    def sma_csdn(close, timeperiod):
        close = np.nan_to_num(close)
        return reduce(lambda x, y: ((timeperiod - 1) * x + y) / timeperiod, close, timeperiod)

    def stock_csdn(high, low, close, fastk_period, slowk_period, slowd_period):
        kValue, dValue = ta.STOCHF(high, low, close, fastk_period, fastd_period=1, fastd_matype=0)

        kValue = np.array(list(map(lambda x: sma_csdn(kValue[:x], slowk_period), range(1, len(kValue) + 1))))
        dValue = np.array(list(map(lambda x: sma_csdn(kValue[:x], slowd_period), range(1, len(dValue) + 1))))
        jValue = 3 * kValue - 2 * dValue

        func = lambda arr: np.array([0 if x < 0 else (100 if x > 100 else x) for x in arr])

        kValue = func(kValue)
        dValue = func(dValue)
        jValue = func(jValue)
        return kValue, dValue

    # 通过STOCK获取K和D
    data_temp["slowk"], data_temp["slowd"] = stock_csdn(
        data_temp["HIGH"].values,
        data_temp["LOW"].values,
        data_temp["CLOSE"].values,
        fastk_period,
        slowk_period,
        slowd_period,
    )
    # 获取J，这里没有用到
    list_slowj = list(map(lambda x, y: 3 * x - 2 * y, data_temp["slowk"], data_temp["slowd"]))
    return data_temp.reset_index()


def get_cmo_idc(df, timeperiod=10, **kwargs):
    column_name = kwargs.get("column_name", "CLOSE")  # 默认使用CLOSE列
    data_temp = (df[column_name].resample("B").last().ffill()).to_frame()  # 确保是DataFrame格式
    data_temp["cmo"] = ta.CMO(data_temp["CLOSE"], timeperiod)
    return data_temp.reset_index()


def get_roc_idc(df, timeperiod=10, **kwargs):
    column_name = kwargs.get("column_name", "CLOSE")  # 默认使用CLOSE列
    data_temp = (df[column_name].resample("B").last().ffill()).to_frame()  # 确保是DataFrame格式
    data_temp["roc"] = ta.ROC(data_temp["CLOSE"].values, timeperiod)
    return data_temp.reset_index()


def get_vol_mom_idc(df, **kwargs):
    column_name = kwargs.get("column_name", ["CLOSE", "VOLUME"])  # 默认使用CLOSE列
    data_temp = df[column_name].resample("B").last().ffill()
    return data_temp.reset_index()


def get_bbands_idc(df, timeperiod=20, nbdevup=2, nbdevdn=2, factor_name="CLOSE"):
    df["upperband"], middleband, df["lowerband"] = ta.BBANDS(
        df[factor_name].values, timeperiod, nbdevup, nbdevdn, matype=0
    )
    return df.reset_index()


# def get_dma_idc(df, short_tp=10, long_tp=20):
#     short_ma = ta.SMA(df["CLOSE"].values, short_tp)
#     long_ma = ta.SMA(df["CLOSE"].values, long_tp)
#     df["dma"] = short_ma - long_ma
#     return df["dma"]


# def get_ama_idc(df, short_tp=10, long_tp=50):
#     short_ma = ta.SMA(df["CLOSE"].values, short_tp)
#     long_ma = ta.SMA(df["CLOSE"].values, long_tp)
#     df["dma"] = short_ma - long_ma
#     df["ama"] = df["dma"].rolling(10).mean()
#     return df["ama"]


# def get_trix_idc(df, timeperiod=12, signalperiod=9):
#     trix = pd.Series(ta.TRIX(df["CLOSE"].values, timeperiod))
#     matrix = trix.rolling(signalperiod).mean()  # rolling 包含当日
#     df["trix_diff"] = (trix - matrix).values  # series to df 要用pd.调用方法，nparray可以直接放进去
#     return df["trix_diff"]


# def get_rsi_idc(df, timeperiod=6, factor_name="CLOSE"):
#     df["rsi"] = ta.RSI(df[factor_name], timeperiod=6)
#     return df["rsi"]


# def get_macd_idc(df, factor_name="CLOSE", fastperiod=12, slowperiod=26, signalperiod=9):
#     df["macd"] = ta.MACD(df[factor_name].values, fastperiod, slowperiod, signalperiod)[2]  # 取hist的
#     return df["macd"]


# def get_obv_idc(df):
#     df["obv"] = 0  # 0为初始值
#     df["pct_change"] = df["close"].pct_change()
#     df["obv"] = (
#         np.where(df["pct_change"] > 0, df["volume"], 0) - np.where(df["pct_change"] < 0, df["volume"], 0)
#     ).cumsum()  # 成交量当净变化的cumsum
#     return df["obv"]


# def get_vspike_idc(df, window=5):
#     df["short_sma"] = df["volume"].rolling(window).mean()
#     df["vol_spike"] = df["volume"] / df["short_sma"]
#     return df["vol_spike"]
