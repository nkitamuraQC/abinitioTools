import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def heatmap_ao(mf, data):
    ao_labels = mf.mol.ao_labels()
    assert data.shape[0] == len(ao_labels)
    assert data.shape[1] == len(ao_labels)

    ao_info = []
    for ao in ao_labels:
        aostr = str(ao[1]) + str(ao[0]) + str(ao[2])
        ao_info.append(aostr)

    df = pd.DataFrame(data, columns=ao_info, index=ao_info)
    sns.heatmap(df, cmap="coolwarm", annot=True)
    plt.show()
    return


def plot_corr(corr):

    plt.plot(corr[0, :])
    plt.show()
    return
