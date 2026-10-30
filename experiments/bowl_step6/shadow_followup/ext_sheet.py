"""Side-by-side of the exact policy inputs (180x320) at rows 8/16/24/48: A2002 vs B, exterior and wrist, plus |diff|."""
import sys,h5py,numpy as np,cv2
ev,sync=sys.argv[1:3]
rows=(8,16,24,48);tiles=[]
with h5py.File(f'{ev}/step6_B_shadow_capture_v1/captures.hdf5') as fb,h5py.File(sync) as fa:
    for r in rows:
        line=[]
        for k in ('video.exterior_image_1_left','video.wrist_image_left'):
            a=fa[k][r];b=fb[str(r)][k][0,0];d=np.clip(np.abs(a.astype(int)-b.astype(int)).sum(-1)*2,0,255).astype(np.uint8)
            d=cv2.applyColorMap(d,cv2.COLORMAP_INFERNO)[...,::-1]
            for im,t in ((a,f'A2002 r{r}'),(b,f'B r{r}'),(d,'|A-B|')):
                im=cv2.resize(np.ascontiguousarray(im),(480,270),interpolation=cv2.INTER_NEAREST).copy()
                cv2.putText(im,f'{t} {k.split(".")[1][:8]}',(6,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,0),1);line.append(im)
        tiles.append(np.hstack(line))
cv2.imwrite('ab_inputs_rows.png',cv2.cvtColor(np.vstack(tiles),cv2.COLOR_RGB2BGR));print('ok')
