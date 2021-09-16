from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial import distance


def calc_knn(df: pd.DataFrame, knn: Dict[int, List[float]], chunk: List[int], vl: int, k: int) -> None:
    for i in chunk:
        c = df.loc[[i]]
        result = df.iloc[:, 0:vl].apply(
            lambda row: distance.euclidean(row, c.iloc[:, 0:vl]), axis=1)
        knn[i] = list(
            dict(sorted(result.to_dict().items(), key=lambda x: x[1])).keys())[1:k+2]


def calc_rknn(df: pd.DataFrame) -> pd.DataFrame:
    rknn = {}
    for i, _ in df['knn'].iteritems():
        rknn[i] = 0
        for _, item_ in df['knn'].iteritems():
            if i in item_:
                rknn[i] += 1
    rknn_l = []
    for i, _ in df['knn'].iteritems():
        rknn_l.append(rknn[i])

    df['rknn'] = pd.Series(rknn_l)
    return df


def calc_ndf(df: pd.DataFrame) -> pd.DataFrame:
    ndf = []
    for _, row in df.iterrows():
        ndf.append(row['rknn']/len(row['knn']))

    df['ndf'] = pd.Series(ndf)
    return df


def calc_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df['cid'] = np.nan

    cluster_count = 0
    for index, row in df.iterrows():
        if not np.isnan(df.loc[index, 'cid']) or row['ndf'] < 1:
            continue
        df.loc[index, 'cid'] = cluster_count
        dpset = []
        for item in row['knn']:
            df.loc[item, 'cid'] = cluster_count
            if df.loc[item, 'ndf'] >= 1:
                dpset.append(item)
        while dpset:
            p = dpset[0]
            for item in df['knn'][p]:
                if not np.isnan(df.loc[item, 'cid']):
                    continue
                df.loc[item, 'cid'] = cluster_count
                if df.loc[item, 'ndf'] >= 1:
                    dpset.append(item)
            dpset.pop(0)
        cluster_count = cluster_count + 1
    return df
