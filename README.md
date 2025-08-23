# sc-utils

Installation:
```bash
pip install sc-utils
```

## cellstate-aligner

Fast, reproducible group-to-group alignment for single-cell datasets with plotting and label transfer.<br>
	•	Vectorized Pearson/Spearman/Cosine for speed<br>
	•	Optional Kendall, distance correlation, or Lin’s concordance (CCC)<br>
	•	Flexible scaling: per-dataset or concat-pseudobulk<br>
	•	Built-in plotting and label transfer<br>
	•	Lightweight, with optional dependencies only when needed

#### Quickstart
```bash
import scanpy as sc
from cellstate_aligner import CellStateAligner

# adata1 = reference, adata2 = query
aligner = CellStateAligner().fit(
    adata1, adata2,
    groupby1="ref_labels",
    groupby2="cluster",
    method="spearman",
    scale=True, scale_mode="concat_pseudobulk",
    min_cells_per_group=5,
)

# visualize similarities
aligner.plot_heatmap(normalize="row_minmax")

# assign labels to query groups and write to adata2
assign_df = aligner.assign(
    strategy="argmax",
    min_score=0.3, min_margin=0.05,
    pred_col="pred_ref", adata_query=adata2
)
assign_df.head()
```

### License
MIT
