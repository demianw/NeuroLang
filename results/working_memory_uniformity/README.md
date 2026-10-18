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

## Comparison with the TF-IDF of the term only

The same uniformity test (whole brain, 2 mm grid, same 3-voxel radius) with the
studies selected by the TF-IDF of the term `working memory` in the
`neurostore` engine (computed from the abstracts; 2,111 studies have a value
above 0.001, 2,031 above 0.03), instead of by SPECTER2:

| selection | map | r with SPECTER2 top 5% | Dice (top 5% voxels) |
|---|---|---|---|
| TF-IDF > 0.001 (2,111) | `uniformity_working_memory_tfidf_gt0.001.nii.gz` | 0.945 | 0.807 |
| TF-IDF > 0.03 = `mentions` (2,031) | `uniformity_working_memory_term_mentions.nii.gz` | 0.945 | 0.807 |
| top 1,794 by TF-IDF | `uniformity_working_memory_tfidf_top1794.nii.gz` | 0.942 | 0.805 |
| top 897 by TF-IDF | `uniformity_working_memory_tfidf_top897.nii.gz` | 0.915 | 0.772 |

Only 497 of the 1,794 SPECTER2 studies (and 233 of the top 897 by TF-IDF) are
among the 2,111 studies with the term, yet the maps agree: the extra studies
activate the same network. `glass_specter2_vs_tfidf.png` shows both maps and
their difference: it stays within -2.4 and +1.9 ratio units, with 2,764
voxels beyond ±1.

## Four approaches: NeuroStore and Neurosynth, TF-IDF and SPECTER2

The same uniformity test on the Neurosynth dataset (14,371 studies, its own
TF-IDF vocabulary; abstracts from PubMed embedded with the same SPECTER2
model and the same Cognitive Atlas definition query), compared with the
NeuroStore maps above.

| approach | studies | map |
|---|---|---|
| NeuroStore, TF-IDF > 0.001 | 2,111 | `..._tfidf_gt0.001` |
| NeuroStore, SPECTER2 top 5% | 1,794 | `..._top5` |
| Neurosynth, TF-IDF > 0.001 | 1,091 | `..._neurosynth_tfidf_gt0.001` |
| Neurosynth, SPECTER2 top 5% | 719 | `..._neurosynth_specter2_top5` |

(`..._neurosynth_tfidf_gt0.03`, `..._neurosynth_tfidf_top5pct`,
`..._neurosynth_tfidf_top2p5pct` and `..._neurosynth_specter2_top2p5` are the
other selections.)

Pearson r (Dice of the top 5% of voxels) between the maps:

| | NeuroStore TF-IDF | NeuroStore SPECTER2 | Neurosynth TF-IDF | Neurosynth SPECTER2 |
|---|---|---|---|---|
| NeuroStore TF-IDF | | 0.945 (0.807) | 0.977 (0.896) | 0.918 (0.782) |
| NeuroStore SPECTER2 | 0.945 (0.807) | | 0.930 (0.796) | 0.957 (0.844) |
| Neurosynth TF-IDF | 0.977 (0.896) | 0.930 (0.796) | | 0.919 (0.789) |
| Neurosynth SPECTER2 | 0.918 (0.782) | 0.957 (0.844) | 0.919 (0.789) | |

The same selection method on the two databases agrees more (0.977 for TF-IDF,
0.957 for SPECTER2) than the two methods on the same database (0.945 and
0.919). The studies of the TF-IDF and SPECTER2 selections barely overlap
(497 of 1,794 in NeuroStore, 233 of 719 in Neurosynth). The strongest voxels
are in the paracingulate gyrus and the insular cortex for all four.
Hippocampus: mean ratio 1.22 to 1.28; its maximum is 2.3 to 2.5 for three
approaches and 2.98 (one voxel above the 2.80 threshold of the map) for
Neurosynth SPECTER2, which has the fewest studies.

Neurosynth and NeuroStore are not independent: both come from the
literature, and their studies overlap (not measured here).

`glass_four_approaches.png` and `agreement_four_approaches.png` show the
maps and the agreement.
