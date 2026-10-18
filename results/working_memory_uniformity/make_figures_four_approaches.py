import numpy as np, nibabel as nib, nilearn.plotting as npl
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ns5=np.load("/tmp/dump_top5.npz",allow_pickle=True); ns2=np.load("/tmp/dump_top2p5.npz",allow_pickle=True); nst=np.load("/tmp/dump_tfidf.npz",allow_pickle=True)
syt=np.load("/tmp/dump_syn_tfidf.npz",allow_pickle=True); sys_=np.load("/tmp/dump_syn_specter.npz",allow_pickle=True); aff=ns5["affine"]
im=lambda d: nib.Nifti1Image(d.astype("float32"),aff)
maps=[("NeuroStore, TF-IDF > 0.001 (2,111 studies)",nst["tfidf_001"]),("NeuroStore, SPECTER2 top 5% (1,794)",ns5["sim_map"]),
      ("Neurosynth, TF-IDF > 0.001 (1,091)",syt["tfidf_001"]),("Neurosynth, SPECTER2 top 5% (719)",sys_["similar5"])]
for fn,d in (("neurosynth_tfidf_gt0.001",syt["tfidf_001"]),("neurosynth_tfidf_gt0.03",syt["tfidf_003"]),("neurosynth_tfidf_top5pct",syt["tfidf_top5pct"]),("neurosynth_tfidf_top2p5pct",syt["tfidf_top2p5pct"]),("neurosynth_specter2_top5",sys_["similar5"]),("neurosynth_specter2_top2p5",sys_["similar2p5"])):
    nib.save(im(d),f"/tmp/viz/uniformity_working_memory_{fn}.nii.gz")
fig,axes=plt.subplots(4,1,figsize=(8,12))
for ax,(t,d) in zip(axes,maps):
    npl.plot_glass_brain(im(d),threshold=2.7,vmax=12,axes=ax,display_mode="ortho",colorbar=True,cmap="hot"); ax.set_title(t,fontsize=10)
fig.suptitle("Uniformity ratio of 'working memory', whole brain, threshold 2.7",fontsize=11)
fig.savefig("/tmp/viz/glass_four_approaches.png",dpi=105,bbox_inches="tight")
# agreement matrix
names=["NeuroStore\nTF-IDF","NeuroStore\nSPECTER2","Neurosynth\nTF-IDF","Neurosynth\nSPECTER2"]
def r(x,y):
    mk=(x>0)|(y>0); return np.corrcoef(x[mk],y[mk])[0,1]
M=np.array([[r(a[1],b[1]) for b in maps] for a in maps])
fig,ax=plt.subplots(figsize=(5,4.2)); h=ax.imshow(M,vmin=0.9,vmax=1,cmap="viridis")
ax.set_xticks(range(4)); ax.set_xticklabels(names,fontsize=8); ax.set_yticks(range(4)); ax.set_yticklabels(names,fontsize=8)
for i in range(4):
    for j in range(4): ax.text(j,i,"%.3f"%M[i,j],ha="center",va="center",color="w" if M[i,j]<0.97 else "k",fontsize=9)
fig.colorbar(h,label="Pearson r"); ax.set_title("Agreement of the uniformity maps",fontsize=10)
fig.savefig("/tmp/viz/agreement_four_approaches.png",dpi=110,bbox_inches="tight"); print("ok")
