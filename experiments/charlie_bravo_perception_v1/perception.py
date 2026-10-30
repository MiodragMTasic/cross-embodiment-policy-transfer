"""Experimental RGB-only masks and calibrated depth backprojection; never a release authority.
No simulator object states, semantic IDs, or scoring labels accepted.
Colour thresholds are specific to this yellow banana/red bowl scene.
"""
import numpy as np
from PIL import Image


def masks(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('native RGB uint8 required')
    hsv = np.asarray(Image.fromarray(rgb).convert('HSV'))
    h,s,v = [hsv[:,:,i].astype(float) for i in range(3)]
    return {'banana': largest_component((h>=30)&(h<=48)&(s>=190)&(v>=110)),
            'bowl': ((h<=12)|(h>=242))&(s>=150)&(v>=35)}


def largest_component(mask):
    """Reject ambiguous similarly sized blobs. No spatial oracle or layout-specific ROI."""
    unseen=mask.copy(); components=[]; height,width=mask.shape
    for y,x in zip(*np.nonzero(mask)):
        if not unseen[y,x]:continue
        unseen[y,x]=False; stack=[(y,x)]; component=[]
        while stack:
            yy,xx=stack.pop();component.append((yy,xx))
            for ny,nx in ((yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)):
                if 0<=ny<height and 0<=nx<width and unseen[ny,nx]:
                    unseen[ny,nx]=False;stack.append((ny,nx))
        if len(component)>=20:components.append(component)
    result=np.zeros_like(mask)
    components.sort(key=len,reverse=True)
    if not components or (len(components)>1 and len(components[1])>.5*len(components[0])):
        return result
    coords=np.asarray(components[0]);result[coords[:,0],coords[:,1]]=True
    return result


def describe_mask(mask):
    y,x = np.nonzero(mask)
    if len(x)<20:
        return {'detected':False,'pixels':int(len(x))}
    return {'detected':True,'pixels':int(len(x)),
            'bbox_xyxy':[int(x.min()),int(y.min()),int(x.max()),int(y.max())],
            'centroid_uv':[float(x.mean()),float(y.mean())]}


def backproject(depth_z, K, mask):
    """Optical-axis depth metres, unresized matching RGB pixels; returns camera XYZ.
    Camera convention: +x right, +y down, +z forward. Not Euclidean ray range.
    """
    d=np.asarray(depth_z,dtype=float); k=np.asarray(K,dtype=float)
    if d.ndim!=2 or mask.shape!=d.shape or k.shape!=(3,3):
        raise ValueError('depth/mask/calibration shape mismatch')
    if not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):
        raise ValueError('invalid pinhole calibration')
    valid=mask & np.isfinite(d) & (d>0) & (d<10)
    y,x=np.nonzero(valid)
    rays=np.linalg.solve(k,np.vstack([x,y,np.ones(len(x))])).T
    return rays*d[y,x,None]


def observe(rgb, depth_z=None, K=None):
    segments=masks(rgb)
    result={'schema':'charlie_perception_draft_v1','source':'RGB colour heuristic',
            'release_authorized':False,'held':'unknown','height_above_rim_m':None,
            'horizontal_offset_m':None,'objects':{n:describe_mask(m) for n,m in segments.items()},
            'limitations':['Masks are unvalidated colour candidates, not semantic ground truth.',
              'Visible surface centroids are not object centres or bowl rim geometry.',
              'No held-state or safe-release inference is implemented.']}
    if depth_z is None or K is None:
        result['metric_status']='unavailable: aligned depth and calibration required'
        return result
    result['metric_status']='surface measurements only; rim fit and uncertainty pending'
    for name,mask in segments.items():
        points=backproject(depth_z,K,mask)
        result['objects'][name]['valid_depth_pixels']=len(points)
        result['objects'][name]['visible_surface_median_camera_xyz_m']=np.median(points,axis=0).tolist() if len(points)>=20 else None
    return result


def finite_points(points, minimum=40):
    p=np.asarray(points,float)
    return p.ndim==2 and p.shape[1]==3 and len(p)>=minimum and np.isfinite(p).all()


def fit_bowl(bowl_world):
    """Exterior-only early unoccluded lip fit. No mesh dimensions are runtime inputs."""
    unknown={'status':'unknown','geometry_allows_request':False}
    b=np.asarray(bowl_world,float)
    if not finite_points(b):return {**unknown,'reason':'insufficient bowl points'}
    ceiling=np.quantile(b[:,2],.99)
    top=b[(b[:,2]>=ceiling-.004)&(b[:,2]<=ceiling+.001)]
    if len(top)<20:return {**unknown,'reason':'insufficient lip band'}
    xy=top[:,:2];ref=xy.mean(axis=0);q=xy-ref
    design=np.column_stack((2*q[:,0],2*q[:,1],np.ones(len(q))))
    if np.linalg.matrix_rank(design)<3:return {**unknown,'reason':'degenerate rim fit'}
    fit=np.linalg.lstsq(design,(q*q).sum(axis=1),rcond=None)[0]
    center=fit[:2]+ref;radius2=fit[2]+sum(fit[:2]**2)
    if radius2<=0:return {**unknown,'reason':'invalid radius'}
    radius=float(np.sqrt(radius2));distances=np.linalg.norm(xy-center,axis=1)
    residual=float(np.quantile(np.abs(distances-radius),.95))
    angles=np.arctan2(xy[:,1]-center[1],xy[:,0]-center[0])
    bins=np.unique(np.minimum(11,((angles+np.pi)/(2*np.pi)*12).astype(int)))
    if not .025<=radius<=.20 or residual>.005 or len(bins)<8:
        return {**unknown,'reason':'rim fit rejected','rim_residual_m':residual,'coverage_bins':len(bins)}
    return {'status':'fitted','bowl_center_xy_m':center.tolist(),'rim_radius_m':radius,
            'rim_height_m':float(np.median(top[:,2])),'rim_residual_m':residual,'coverage_bins':len(bins)}


def measured_gate(banana_world, bowl_world, reference):
    """Visible-surface centroid veto. NOT a held-state estimate or proof of safe release."""
    unknown={'status':'unknown','geometry_allows_request':False}
    a=np.asarray(banana_world,float);b=np.asarray(bowl_world,float)
    if reference.get('status')!='fitted' or not finite_points(a) or not finite_points(b):
        return {**unknown,'reason':'missing calibration or visible surfaces'}
    # Compare current visible red points to the early, camera-measured surface.
    # Stored surface includes wall/lip; no full wrist rim fit is required.
    surface=np.asarray(reference['surface_world_m'])
    sampled=b[::max(1,len(b)//400)]
    distances=[]
    for part in np.array_split(sampled, max(1,len(sampled)//64)):
        distances.extend(np.sqrt(((part[:,None,:]-surface[None,:,:])**2).sum(2).min(1)))
    mismatch=float(np.median(distances))
    center=np.asarray(reference['bowl_center_xy_m'])
    offset=float(np.linalg.norm(np.median(a[:,:2],axis=0)-center))
    low=float(np.quantile(a[:,2],.05)-reference['rim_height_m'])
    allowed=bool(mismatch<=.005 and offset<=.03 and .02<=low<=.10)
    return {**{k:v for k,v in reference.items() if k!='surface_world_m'},
        'status':'draft_measurement','geometry_allows_request':allowed,
        'bowl_surface_mismatch_m':mismatch,'bowl_still_matches':mismatch<=.005,
        'banana_surface_offset_m':offset,'banana_low_surface_height_above_rim_m':low,
        'banana_surface_median_world_m':np.median(a,axis=0).tolist(),
        'banana_points':len(a),'bowl_points':len(b),'held':'unknown'}


class BowlReference:
    def __init__(self):self.reference=None

    def measure(self,tick,banana_world,exterior_bowl_world,calibration_world=None):
        # Lock the first acceptable exterior fit in the declared initial window.
        if self.reference is None and 0<=tick<=60:
            fit=fit_bowl(exterior_bowl_world if calibration_world is None else calibration_world)
            if fit['status']=='fitted':
                b=np.asarray(exterior_bowl_world if calibration_world is None else calibration_world)
                # Bounded, evenly sampled observed surface, never object state.
                fit['surface_world_m']=b[np.linspace(0,len(b)-1,min(4000,len(b)),dtype=int)].tolist()
                fit['calibration_tick']=tick
                self.reference=fit
        if self.reference is None:return {'status':'unknown','geometry_allows_request':False,'reason':'no early exterior calibration'}
        return measured_gate(banana_world,exterior_bowl_world,self.reference)


def geometry_evidence(banana_world,bowl_world):
    """One-frame development helper; live capture uses the persistent BowlReference."""
    return BowlReference().measure(0,banana_world,bowl_world)


def camera_to_world(points, world_from_optical):
    """Explicit calibrated +x right,+y down,+z forward transform; never guess convention."""
    p=np.asarray(points,float);t=np.asarray(world_from_optical,float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):
        raise ValueError('invalid optical-to-world transform')
    r=t[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1,atol=1e-5):
        raise ValueError('transform rotation must be proper orthonormal')
    return p@r.T+t[:3,3]
