import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_ao(mf, data):
    ao_labels = mf.mol.ao_labels()
    assert(data.shape[0] == len(ao_labels))
    assert(data.shape[1] == len(ao_labels))

    df = pd.DataFrame(data, columns=ao_labels, index=ao_labels)
    sns.heatmap(df, cmap='coolwarm', annot=True)
    plt.show()
    return
  
