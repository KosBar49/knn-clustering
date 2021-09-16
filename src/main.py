import argparse
import configparser
import pathlib
from multiprocessing import Manager, Process

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs

from calcs.calcs import *

SAMPLE = "sample.csv"
CONFIG = "config.ini"


def get_root() -> str:
    return str(pathlib.Path(__file__).parent.resolve())


def read_sample(filename: str) -> pd.DataFrame:
    root_ = get_root()
    df = pd.read_csv(root_ + "/data/" + filename)
    return df


def read_config(filename: str):
    config = configparser.ConfigParser()
    config.read(get_root() + "/config/" + filename)
    default = config['DEFAULT']
    return (int(default.get("DataLength")), int(default.get("Processes")),
            int(default.get("VectorLength")), int(default.get("Neighboorhod")))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str,
                        choices=['file', 'random'],
                        help="Choose the mode of data source used for script run")
    args = parser.parse_args()
    dl, n, vl, k = read_config(CONFIG)

    if args.mode == "random":
        data, _ = make_blobs(n_samples=dl, n_features=vl, centers=2)
        data -= data.min(axis=0)
        df = pd.DataFrame(data)
    elif args.mode == "file":
        df = read_sample(SAMPLE)

    # Chunk data for multiprocessing
    chunks = np.array_split(df.index.values, n)
    manager = Manager()
    knn = manager.dict()

    # Calculate knn
    for chunk in chunks:
        p = Process(target=calc_knn, args=(df, knn, chunk, vl, k,))
        p.start()
        p.join()

    df['knn'] = pd.Series(knn)

    # Calculate rknn
    df = calc_rknn(df)

    # Calculate NDF
    df = calc_ndf(df)

    # Calculate clusters
    df = calc_clusters(df)

    # Show results of clustering
    print(df)

    if vl == 2:
        fig, ax = plt.subplots()
        ax.scatter(df.iloc[:,0], df.iloc[:,1])

        for i, txt in enumerate(df.loc[:,'cid']):
            ax.annotate(txt, (df.iloc[i,0], df.iloc[i,1]))
        plt.grid(True)
        plt.title('{}-KNN Clustering'.format(k))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig('results.png')
