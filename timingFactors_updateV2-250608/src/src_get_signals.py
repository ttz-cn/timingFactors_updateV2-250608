import pandas as pd
import numpy as np
import talib as ta


def positional_rule_reference_line(df, indicator_name, reference_line, threshold=0):
    diff = df[reference_line] - df[indicator_name]
    df.loc[:, "signal"] = np.where(diff > threshold, 1, np.where(diff < -threshold, -1, 0))
    return df.loc[:, "signal"]


def cross_rule(df, isValue=True, **kargs):
    def cross_rule_value(indicator_name, value1, value2):  # 传入value1、value2
        con_1 = ((df[indicator_name].shift(1) < value1) & (df[indicator_name] > value1)) | (
            (df[indicator_name].shift(1) < value2) & (df[indicator_name] > value2)
        )
        con_2 = ((df[indicator_name].shift(1) > value1) & (df[indicator_name] < value1)) | (
            (df[indicator_name].shift(1) > value2) & (df[indicator_name] < value2)
        )
        df.loc[:, "signal"] = np.where(con_1, 1, np.where(con_2, -1, 0))

    def cross_rule_reference_line(ReferenceLine1, ReferenceLine2):  # 传入ReferenceLine1，ReferenceLine2
        con_1 = (df[ReferenceLine1].shift(1) < df[ReferenceLine2].shift(1)) & (
            df[ReferenceLine1] > df[ReferenceLine2]
        )  # 上穿买入，动量策略
        con_2 = (df[ReferenceLine1].shift(1) > df[ReferenceLine2].shift(1)) & (df[ReferenceLine1] < df[ReferenceLine2])
        df.loc[:, "signal"] = np.where(con_1, 1, np.where(con_2, -1, 0))

    if isValue:
        cross_rule_value(indicator_name=kargs["indicator_name"], value1=kargs["value1"], value2=kargs["value2"])
        return df.loc[:, "signal"]

    cross_rule_reference_line(ReferenceLine1=kargs["ReferenceLine1"], ReferenceLine2=kargs["ReferenceLine2"])
    return df.loc[:, "signal"]


def positional_rule_bands(df, upperband, lowerband, reference_line, isMTM=True):
    con_1 = df[reference_line] > df[upperband]
    con_2 = df[reference_line] < df[lowerband]

    if isMTM:
        df.loc[:, "signal"] = np.where(con_1, 1, np.where(con_2, -1, 0))
    else:
        df.loc[:, "signal"] = np.where(con_2, 1, np.where(con_1, -1, 0))

    return df.loc[:, "signal"]


def mom_rule(df, indicator_name="CLOSE", isMTM=True):
    if isMTM:
        con_1 = df[indicator_name] > df[indicator_name].shift(1)
        con_2 = df[indicator_name] < df[indicator_name].shift(1)
    else:
        con_1 = df[indicator_name] < df[indicator_name].shift(1)
        con_2 = df[indicator_name] > df[indicator_name].shift(1)

    df.loc[:, "signal"] = np.where(con_1, 1, np.where(con_2, -1, 0))
    return df.loc[:, "signal"]


def quantile_rule(df, rolling_window, threshold=0.3):
    df["quantile"] = df["CLOSE"].rolling(rolling_window).apply(lambda series: series.rank(pct=True).iloc[-1], raw=False)
    df.loc[:, "signal"] = np.where(df["quantile"] < threshold, 1, np.where(df["quantile"] > 1 - threshold, -1, 0))
    return df.loc[:, "signal"]


def positional_rule_value(df, indicator_name, isInverse=False, value1=0, value2=0):
    if isInverse:
        con_1 = df[indicator_name] < value1
        con_2 = df[indicator_name] > value2
    else:
        con_1 = df[indicator_name] > value1
        con_2 = df[indicator_name] < value2
    df.loc[:, "signal"] = np.where(con_1, 1, np.where(con_2, -1, 0))
    return df.loc[:, "signal"]


def dma_ama_rule(df):
    # 根据sma判断买入卖出信号并写入（当天信号。当天卖出）
    con_1 = (
        (df["dma"] > 0)
        & (df["ama"] > 0)
        & ((df["dma"] - df["dma"].shift(1)) > 0)
        & ((df["ama"] - df["ama"].shift(1)) > 0)
    )
    con_2 = (
        (df["dma"] < 0)
        & (df["ama"] < 0)
        & ((df["dma"] - df["dma"].shift(1)) < 0)
        & ((df["ama"] - df["ama"].shift(1)) < 0)
    )
    df.loc[:, "signal"] = np.where(con_1, 1, np.where(con_2, -1, 0))
    return df.loc[:, "signal"]


def consecutive_days_rule(df, indicator_name, days=3):
    df["return"] = df[indicator_name].pct_change(1)
    df["streak_up"] = (
        df["return"].gt(0).astype(int).groupby(df["return"].le(0).astype(int).cumsum()).cumsum()
    )  # gt(0)=1计数，gt(0)=0分组
    df["streak_down"] = df["return"].lt(0).astype(int).groupby(df["return"].ge(0).astype(int).cumsum()).cumsum()

    con_1 = df["streak_up"] >= days
    con_2 = df["streak_down"] >= days
    df.loc[:, "signal"] = np.where(con_1, 1, np.where(con_2, -1, 0))
    return df.loc[:, "signal"]


def yoy_mtm_rule(df, indicator_name, shift_p=12):
    df["yoy"] = df[indicator_name] / df[indicator_name].shift(shift_p)
    con_1 = df["yoy"] > df["yoy"].shift(1)
    con_2 = df["yoy"] < df["yoy"].shift(1)
    df.loc[:, "signal"] = np.where(con_1, 1, np.where(con_2, -1, 0))
    return df.loc[:, "signal"]


def volume_price_div_rule(df, timeperiod=5):

    df["price_roc"] = ta.ROC(df["close"].values, timeperiod)
    df["vol_roc"] = ta.ROC(df["volume"].values, timeperiod)

    df.loc[:, "signal"] = np.where(
        (df["price_roc"] < 0) & (df["vol_roc"] > 0), 1, np.where((df["price_roc"] > 0) & (df["vol_roc"] < 0), -1, 0)
    )
    return df.loc[:, "signal"]


def volume_price_momentum_rule(df, indicator_name=["CLOSE", "VOLUME"]):

    df["price_change"] = df[indicator_name[0]].pct_change()
    df["volume_change"] = df[indicator_name[1]].pct_change()
    df.loc[:, "signal"] = np.where(
        (df["price_change"] > 0) & (df["volume_change"] > 0),
        1,
        np.where((df["price_change"] < 0) & (df["volume_change"] < 0), -1, 0),
    )
    return df.loc[:, "signal"]
