import numpy as np, nibabel as nib, nilearn.plotting as npl, nilearn.datasets as nd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a=np.load("/tmp/dump_top5.npz",allow_pickle=True); t=np.load("/tmp/dump_tfidf.npz",allow_pickle=True); aff=a["affine"]
specter=a["sim_map"]; tf=t["tfidf_001"]
im=lambda d: nib.Nifti1Image(d.astype("float32"),aff)
nib.save(im(tf),"/tmp/viz/uniformity_working_memory_tfidf_gt0.001.nii.gz")
nib.save(im(t["tfidf_top1794"]),"/tmp/viz/uniformity_working_memory_tfidf_top1794.nii.gz")
nib.save(im(t["tfidf_top897"]),"/tmp/viz/uniformity_working_memory_tfidf_top897.nii.gz")
fig,axes=plt.subplots(3,1,figsize=(8,9))
npl.plot_glass_brain(im(specter),threshold=2.66,vmax=12,axes=axes[0],display_mode="ortho",colorbar=True,cmap="hot"); axes[0].set_title("SPECTER2: top 5% by similarity to the Cognitive Atlas definition (1,794 studies)",fontsize=9)
npl.plot_glass_brain(im(tf),threshold=2.70,vmax=12,axes=axes[1],display_mode="ortho",colorbar=True,cmap="hot"); axes[1].set_title("TF-IDF of 'working memory' > 0.001 (2,111 studies)",fontsize=9)
diff=specter-tf; brain=(specter>0)|(tf>0)
npl.plot_glass_brain(im(np.where(brain,diff,0)),threshold=1.0,vmax=4,axes=axes[2],display_mode="ortho",colorbar=True,cmap="RdBu_r",plot_abs=False); axes[2].set_title("SPECTER2 minus TF-IDF ratio (red: higher with SPECTER2; |difference| > 1)",fontsize=9)
fig.savefig("/tmp/viz/glass_specter2_vs_tfidf.png",dpi=110,bbox_inches="tight")
d=np.where(brain,diff,0); print("diff: max %.2f min %.2f | voxels > +1: %d, < -1: %d" % (d.max(),d.min(),(d>1).sum(),(d<-1).sum()))
