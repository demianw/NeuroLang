import numpy as np, nibabel as nib, nilearn.datasets as nd, nilearn.image as ni
ns5=np.load("/tmp/dump_top5.npz",allow_pickle=True); nst=np.load("/tmp/dump_tfidf.npz",allow_pickle=True)
sy_t=np.load("/tmp/dump_syn_tfidf.npz",allow_pickle=True); sy_s=np.load("/tmp/dump_syn_specter.npz",allow_pickle=True)
aff=ns5["affine"]
A={"NeuroStore TF-IDF>0.001":(nst["tfidf_001"],set(map(str,nst["tfidf_001_studies"]))),
   "NeuroStore SPECTER2 5%":(ns5["sim_map"],set(map(str,ns5["similar"]))),
   "Neurosynth TF-IDF>0.001":(sy_t["tfidf_001"],set(map(str,sy_t["tfidf_001_studies"]))),
   "Neurosynth SPECTER2 5%":(sy_s["similar5"],set(map(str,sy_s["similar5_studies"])))}
names=list(A)
for n,(m,s) in A.items(): print("%-26s studies %5d | max %5.1f | nonzero voxels %d" % (n,len(s),m.max(),(m>0).sum()))
def agree(x,y):
    mk=(x>0)|(y>0); u,v=x[mk],y[mk]; ta,tb=u>np.quantile(u,.95),v>np.quantile(v,.95)
    return np.corrcoef(u,v)[0,1], 2*(ta&tb).sum()/(ta.sum()+tb.sum())
print("\npairwise  r / Dice(top 5% voxels):")
print("%-26s" % "" + "".join("%-26s" % n for n in names))
for a in names:
    print("%-26s" % a + "".join("%-26s" % ("-" if a==b else "%.3f / %.3f" % agree(A[a][0],A[b][0])) for b in names))
print("\nstudy overlap within a dataset:")
print("  NeuroStore: TF-IDF %d, SPECTER2 %d, both %d" % (len(A[names[0]][1]),len(A[names[1]][1]),len(A[names[0]][1]&A[names[1]][1])))
print("  Neurosynth: TF-IDF %d, SPECTER2 %d, both %d" % (len(A[names[2]][1]),len(A[names[3]][1]),len(A[names[2]][1]&A[names[3]][1])))
grid=nib.Nifti1Image(np.zeros(ns5["sim_map"].shape),aff)
ho=nd.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm",data_dir="/tmp/hodata",verbose=0)
lab=np.asarray(ni.resample_to_img(nib.load(ho.maps) if isinstance(ho.maps,str) else ho.maps,grid,interpolation="nearest").dataobj).astype(int)
groups={"Hippocampus":["Hippocampus"],"Amygdala":["Amygdala"],"Thalamus":["Thalamus"],"Caud+Put+Pall":["Caudate","Putamen","Pallidum"]}
ids={g:[i for i,n in enumerate(ho.labels) if any(k in n for k in ks)] for g,ks in groups.items()}
print("\nmean / max ratio per structure (* = a voxel above the map's own 95th-percentile threshold):")
print("%-26s %6s | " % ("map","thr")+" | ".join("%-15s" % g for g in groups))
for n,(m,_) in A.items():
    thr=np.quantile(m[m>0],.95); cells=[]
    for g in groups:
        k=np.isin(lab,ids[g]); cells.append("%.2f / %.2f %s" % (m[k].mean(),m[k].max(),"*" if (m[k]>thr).any() else " "))
    print("%-26s %6.2f | " % (n,thr)+" | ".join("%-15s" % c for c in cells))
# top regions by label (cortex included) for each map: where the maxima are
cort=nd.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm",data_dir="/tmp/hodata",verbose=0)
cl=np.asarray(ni.resample_to_img(nib.load(cort.maps) if isinstance(cort.maps,str) else cort.maps,grid,interpolation="nearest").dataobj).astype(int)
print("\nregion of the 20 largest-ratio voxels (cortical Harvard-Oxford labels), per map:")
from collections import Counter
for n,(m,_) in A.items():
    top=np.argsort(m.ravel())[::-1][:300]; c=Counter(cort.labels[k] for k in cl.ravel()[top] if k>0)
    print("  %-26s %s" % (n, ", ".join("%s %d" % (k[:26],v) for k,v in c.most_common(5))))
