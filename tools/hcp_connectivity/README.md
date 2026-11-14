# The structural and functional connectivity of the engines, from the HCP

This package makes the dataset
https://gitlab.inria.fr/dwasserm/neurolang-hcp-connectivity (see its
README for what it holds and how to acknowledge it), whose tables the
NeuroLang engines download (the `relations` of `structural_connectivity`
and `functional_connectivity`, at a tag, with their md5):

- the **structural** connectivity: the high-resolution structural
  connectomes of Mansour et al. (2021), one 59,412 × 59,412 sparse matrix
  per subject of the HCP Young Adult study (1,064 of them, 557 GB), in the
  authors' public bucket (https://github.com/sina-mansour/neural-identity);
- the **functional** connectivity: the resting-state fMRI of the HCP
  subjects with the four runs of rest (ICA-FIX cleaned, MSMAll, 91,282
  grayordinates; HCP open access data, which you obtain from the HCP).

Run it from `tools/` (or with `tools/` on `PYTHONPATH`), with a Python of
numpy, scipy, pandas, pyarrow, nibabel and nilearn:

```sh
python -m hcp_connectivity atlases WORK          # needs the internet
```

writes the parcel of each of the 59,412 cortical vertices in Schaefer 2018
(100 to 1000 parcels, 7 and 17 networks), Glasser 2016 and Destrieux 2009
in `WORK/atlases.npz` (`atlases.py`); Tian's subcortex (scale 2) is mapped
onto the subcortical grayordinates of the data when they are read.

The workers sum the subjects' matrices of each atlas, and their squares,
into parts (`OUT/part-<task>.npz`, saved as they go, so that a worker
started again goes on where it stopped); `combine` makes the group's
mean and standard deviation, the engines' tables and the gradients.
`--task I --tasks N` give a worker its share; by default those of a Slurm
array, so on a cluster:

```sh
sbatch --array=0-39 --mem=12G --wrap \
  "python -m hcp_connectivity functional WORK HCP_DIR PARTS"
```

and on one machine, a loop of `--task` (in parallel as memory allows):

```sh
for i in $(seq 0 7); do
  python -m hcp_connectivity functional WORK HCP_DIR PARTS --task $i --tasks 8 &
done; wait
```

## Structural

1. Workers that parcellate the connectomes as they come into `STAGING`
   (a scratch directory: about 500 MB each, deleted once parcellated),
   any number of them, each its own `--task`:
   `python -m hcp_connectivity structural WORK STAGING PARTS --task I`.
2. The connectomes, into `STAGING`, by one of:
   - `python -m hcp_connectivity download STAGING PARTS` where the
     internet is reachable;
   - `relay.sh HOST STAGING PARTS LOCAL_DIR`, from another machine, when
     the cluster's own downloads are slow: each file through `LOCAL_DIR`,
     copied with scp (`SSH_OPTIONS` for ssh and scp).

   Both skip the subjects of the parts, and end with `STAGING/DONE`,
   which ends the workers.
3. `python -m hcp_connectivity combine structural WORK PARTS DATASET`:
   the mean and standard deviation matrices of each atlas
   (`parquet/<atlas>_mean.parquet`, `<atlas>_sd.parquet`, their columns
   and index the regions), the subjects, a TSV per atlas, the gradients,
   and `validation.tsv` (our Glasser matrices against the authors',
   subject by subject).

## Functional

1. `python -m hcp_connectivity functional WORK HCP_DIR PARTS --task I
   --tasks N`: for each subject, each run's time series averaged within
   each region (the cortical parcels of an atlas, then Tian's subcortex),
   their correlations, Fisher z, averaged over the runs.
2. `python -m hcp_connectivity combine functional WORK PARTS DATASET`:
   the correlation of the mean z (`parquet/<atlas>_functional_mean.parquet`,
   over the parcels then Tian's regions), the standard deviation of z,
   the engines' tables (two parcels, a parcel and a Tian region, two Tian
   regions) and the gradients.

## Tractography

The structural connectivity of each subject from its own tractography
(`tractography.py`), over Schaefer's 400 parcels (17 networks) and the 32
regions of Tian's subcortex (scale 2): 432 × 432. It needs MRtrix3 (3.0.4
tested), the HCP's preprocessed diffusion data in the subject's T1w space
(`T1w/Diffusion/{data.nii.gz,bvals,bvecs,nodif_brain_mask.nii.gz}`), its
`T1w/aparc+aseg.nii.gz` and the warp `MNINonLinear/xfms/standard2acpc_dc`;
no FSL.

1. `python -m hcp_connectivity volume-atlas WORK` (needs the internet):
   Schaefer's volume and Tian's atlas in MNI space, combined (cortex
   1-400, Tian 401-432) in `WORK/atlases`; `WORK` is also where
   `atlases.npz` (above) names the parcels.
2. `python -m hcp_connectivity tractography WORK HCP_DIR OUT --task I
   --tasks N [--limit 100] [--streamlines 5000000] [--scratch DIR]`, a task
   of the subjects that have diffusion data (the first `--limit`, sorted).
   The MRtrix3 commands are run by `$MRTRIX_CMD`: for a container,
   `MRTRIX_CMD="singularity exec mrtrix3.sif"` (with the folders of the
   data bound, `SINGULARITY_BIND`); empty, those on the `PATH`. The threads
   are `$SLURM_CPUS_PER_TASK`, or all the CPUs. For each subject, in
   `--scratch` (default `WORK/tractography`): `dwi2response dhollander`,
   `dwi2fod msmt_csd` (white matter and CSF), `5ttgen freesurfer
   -sgm_amyg_hipp`, `tckgen` (iFOD2, ACT, backtracking, cropped at the
   grey/white interface, dynamic seeding, 5 million streamlines), `tcksift2`,
   the atlas brought to the subject through the HCP's warp (nearest
   neighbour, `nibabel`/`scipy`, no FSL) and checked against its
   segmentation (Dice of Tian's structures and aseg's), and
   `tck2connectome -symmetric -zero_diagonal -assignment_radial_search 2`.
   The steps done are marked, so that a task started again goes on after
   the last one; the intermediate files, the tractogram included, are
   deleted once the matrix is made (`--keep` to keep them).
3. The outputs, in `OUT`: `part-<task>.npz` (as above; the sum of the
   subjects' matrices of `schaefer400_17_tian`, the SIFT2 weights between
   two regions times SIFT2's proportionality coefficient so that the
   subjects are comparable, and of `schaefer400_17_tian_count`, the
   streamlines), `matrices/<subject>.npz` (the two matrices of the
   subject), `qc/<subject>.json` (the time of each step, the streamlines
   with both ends assigned, the atlas's Dice, and what the matrix should
   show: the thalamus to the cortex, the striatum to the frontal cortex,
   the hippocampus to the temporal, the interhemispheric share).
4. `python -m hcp_connectivity combine tractography WORK PARTS DATASET`:
   the mean and standard deviation of both matrices
   (`parquet/schaefer400_17_tian[_count]_{mean,sd}.parquet`), the subjects
   (`tractography_subjects.parquet`) and the tables of the engines
   (`tsv/..._tian.tsv.gz`: a parcel and a Tian region; `..._tian_tian`;
   the parcels between them).

## Gradients

`gradients.py`: a principal component analysis of the regions'
connectivity profiles within each hemisphere, the right hemisphere's
rotated onto the left's (see its docstring);
`python -m hcp_connectivity gradients WORK DATASET` makes them again from
the matrices of a dataset.

A new version of the dataset is a new tag there: the engines' URLs and
md5s name it. The TSVs are gzipped without their time, so that a file
made again from the same data is the same.
