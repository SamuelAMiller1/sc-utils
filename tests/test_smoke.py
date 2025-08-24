import numpy as np
import pandas as pd
import anndata as ad
from sc_utils import CellStateAligner

def _toy(n_cells=40, n_genes=20):
    X = np.random.RandomState(0).randn(n_cells, n_genes)
    obs = pd.DataFrame({"grp": np.repeat(["a","b","c","d"], n_cells//4)})
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)

def test_fit_and_assign():
    ad1 = _toy()
    ad2 = _toy()
    aligner = CellStateAligner().fit(ad1, ad2, groupby1="grp", groupby2="grp", method="spearman", scale=False)
    df = aligner.assign(strategy="argmax")
    assert "pred_ref" in df.columns
