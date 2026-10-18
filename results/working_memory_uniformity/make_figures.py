import numpy as np, nibabel as nib, nilearn.plotting as npl, nilearn.datasets as nd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a=np.load("/tmp/dump_top5.npz",allow_pickle=True); b=np.load("/tmp/dump_top2p5.npz",allow_pickle=True)
aff=a["affine"]
imgs={"top5":nib.Nifti1Image(a["sim_map"].astype("float32"),aff),
      "top2p5":nib.Nifti1Image(b["sim_map"].astype("float32"),aff),
      "term_mentions":nib.Nifti1Image(a["term_map"].astype("float32"),aff)}
for k,v in imgs.items(): nib.save(v,f"/tmp/viz/uniformity_working_memory_{k}.nii.gz")
thr=2.66; vmax=11.6
titles={"top5":"top 5% studies by similarity to the\nCognitive Atlas definition (1,794 studies)",
        "top2p5":"top 2.5% (897 studies)",
        "term_mentions":"studies mentioning 'working memory'\n(2,031 studies)"}
# 1. glass brains, same threshold and colour scale
fig,axes=plt.subplots(3,1,figsize=(8,9))
for ax,(k,v) in zip(axes,imgs.items()):
    npl.plot_glass_brain(v,threshold=thr,vmax=vmax,axes=ax,display_mode="ortho",colorbar=True,cmap="hot")
    ax.set_title(titles[k],fontsize=10)
fig.suptitle("Uniformity ratio P(A_v|T)/P(A_V|T), whole brain, threshold %.2f" % thr,fontsize=11)
fig.savefig("/tmp/viz/glass_5_vs_2p5.png",dpi=110,bbox_inches="tight"); plt.close(fig)
# 2. slices through the hippocampus, thalamus/basal ganglia (same threshold; 1.5 shown lower to see sub-threshold hippocampus)
bg=nd.load_mni152_template(resolution=2)
for lo,name in ((thr,"thr2p66"),(1.5,"thr1p5")):
    fig,axes=plt.subplots(3,1,figsize=(11,9))
    for ax,(k,v) in zip(axes,imgs.items()):
        npl.plot_stat_map(v,bg_img=bg,threshold=lo,vmax=vmax,axes=ax,display_mode="z",cut_coords=[-18,-12,-6,0,6,12],colorbar=True,cmap="hot",title=titles[k].replace("\n"," "),annotate=True,draw_cross=False)
    fig.suptitle("Axial slices (z = -18..12 mm: hippocampus / amygdala at -18..-6, basal ganglia and thalamus above), threshold %.2f" % lo,fontsize=10)
    fig.savefig(f"/tmp/viz/slices_hippocampus_{name}.png",dpi=100,bbox_inches="tight"); plt.close(fig)
print("ok")
