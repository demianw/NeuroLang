# Uniformity maps of "working memory", linked to the NeuroStore studies with SPECTER2

Produced with `examples/plot_cognitive_atlas_uniformity.py` of
[NeuroLang PR #958](https://github.com/NeuroLang/NeuroLang/pull/958) on the
NeuroStore 2026-10 release (36,884 studies, 35,880 with an abstract), the
`neurostore` engine, the 2 mm MNI grid (99x117x95), whole brain.

The Cognitive Atlas definition of *working memory* is embedded with SPECTER2;
the studies of the top 5% (1,794) or 2.5% (897) of the cosine similarities of
their title and abstract to it are the studies of the term. Each map is the
uniformity ratio P(A_v | T) / P(A_V | T): the probability that such a study
activates the voxel v (a peak within 3 voxels), over its average across the
brain; 1 is no effect.

| file | selection |
|---|---|
| `uniformity_working_memory_top5.nii.gz` | top 5% by similarity (1,794 studies, 478 mention the term) |
| `uniformity_working_memory_top2p5.nii.gz` | top 2.5% (897 studies, 289 mention the term) |
| `uniformity_working_memory_term_mentions.nii.gz` | the 2,031 studies that mention "working memory" (TF-IDF > 0.03) |

Figures: `glass_5_vs_2p5.png` (glass brains), `slices_hippocampus_thr2p66.png`
and `slices_hippocampus_thr1p5.png` (axial slices, z = -18 to 12 mm, at the
threshold of the 95th percentile of the map and at 1.5).

Agreement with the term map: r = 0.945 / 0.936 (top 5% / 2.5%), Dice of the
top 5% of voxels 0.807 / 0.807; the two selections' maps correlate at 0.98.
Hippocampus: mean ratio 1.13-1.32, maximum 2.4-2.8, below the 2.66 threshold
(the basal ganglia and the thalamus reach 3.4-4.5). The numbers were checked
against an independent numpy computation of the same selection.
