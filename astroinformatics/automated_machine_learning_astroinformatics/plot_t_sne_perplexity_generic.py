"""
=============================================================================
 t-SNE: The effect of various perplexity values on the shape
=============================================================================

An illustration of t-SNE on the two concentric circles and the S-curve
datasets for different perplexity values.

We observe a tendency towards clearer shapes as the preplexity value increases.

The size, the distance and the shape of clusters may vary upon initialization,
perplexity values and does not always convey a meaning.

As shown below, t-SNE for higher perplexities finds meaningful topology of
two concentric circles, however the size and the distance of the circles varies
slightly from the original. Contrary to the two circles dataset, the shapes
visually diverge from S-curve topology on the S-curve dataset even for
larger perplexity values.

For further details, "How to Use t-SNE Effectively"
http://distill.pub/2016/misread-tsne/ provides a good discussion of the
effects of various parameters, as well as interactive plots to explore
those effects.
"""

# Author: Narine Kokhlikyan <narine@slice.com>
# License: BSD

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time

import pandas as pd
import pdb

def function_tsne_generic(str_file_name):
    """

    Returns:

    """

    n_samples = 300
    n_components = 2
    (fig, subplots) = plt.subplots(3, 5, figsize=(15, 8))
    perplexities = [5, 30, 50, 100]

    #X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    #str_file_name = "breast-cancer-wisconsin_MOD.data"
    X_withcode = pd.read_csv(str_file_name)#, header=None)

    # modifications here and data munging
    X = X_withcode.iloc[:,1:] # ignore code column
    #pdb.set_trace()

    for i, perplexity in enumerate(perplexities):
        ax = subplots[1][i + 1]

        t0 = time()
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(X)
        t1 = time()
        print("S-curve, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[:, 0], Y[:, 1],  cmap=plt.cm.viridis) # c=color,
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')



    plt.show()

    #pdb.set_trace()



function_tsne_generic(str_file_name="breast-cancer-wisconsin_MOD.data")