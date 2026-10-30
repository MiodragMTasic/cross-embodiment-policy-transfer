"""Bounded local damped IK. CPU success does not establish collision-free motion."""
import numpy as np
from scipy.spatial.transform import Rotation
from rigid_chain import rigid_inverse


def pose_error(target,current):
    return np.r_[target[:3,3]-current[:3,3],Rotation.from_matrix(target[:3,:3]@current[:3,:3].T).as_rotvec()]


def physical_limits(graph):
    named={j['name']:j for j in graph['joints']}
    return np.array([[np.deg2rad(named[n]['lower_limit_degrees']) if named[n]['lower_limit_degrees'] is not None else -np.inf,
        np.deg2rad(named[n]['upper_limit_degrees']) if named[n]['upper_limit_degrees'] is not None else np.inf]
        for n in graph['arm_joint_names']])


def solve(chain,target,seed,limits,tool_offset=None,max_iterations=200):
    rigid_inverse(target)
    seed=np.asarray(seed,dtype=float);limits=np.asarray(limits,dtype=float)
    if seed.shape!=(len(chain.names),) or limits.shape!=(len(seed),2) or not np.isfinite(seed).all():
        raise ValueError('Invalid seed/limits')
    if np.isnan(limits).any() or np.any(limits[:,0]>=limits[:,1]) or np.any(seed<limits[:,0]) or np.any(seed>limits[:,1]):
        raise ValueError('Seed outside physical limits')
    if not isinstance(max_iterations,int) or not 1<=max_iterations<=200:raise ValueError('Invalid iteration cap')
    # Artificial local branch window only on unbounded sides; never call it a physical limit.
    lower=np.where(np.isfinite(limits[:,0]),limits[:,0],seed-np.pi)
    upper=np.where(np.isfinite(limits[:,1]),limits[:,1],seed+np.pi)
    q=seed.copy();history=[];weights=np.array([1.,1.,1.,.2,.2,.2]);evaluations=0
    reason='iteration_limit';success=False
    for iteration in range(max_iterations):
        t,j=chain.fk_jacobian(q,tool_offset);evaluations+=1;e=pose_error(target,t)
        position=float(np.linalg.norm(e[:3]));angle=float(np.linalg.norm(e[3:]))
        history.append({'iteration':iteration,'position_error_m':position,'rotation_error_rad':angle,'q':q.tolist()})
        if position<=1e-4 and angle<=1e-3:
            success=True;reason='converged';break
        jw=weights[:,None]*j;ew=weights*e
        delta=jw.T@np.linalg.solve(jw@jw.T+(.002**2)*np.eye(6),ew)
        if not np.isfinite(delta).all():raise ValueError('Nonfinite IK update')
        delta*=min(1.,.15/max(float(np.max(np.abs(delta))),1e-15))
        cost=float(ew@ew);accepted=False
        for scale in (1.,.5,.25,.125,.0625):
            candidate=np.clip(q+scale*delta,lower,upper)
            trial=pose_error(target,chain.fk(candidate,tool_offset));evaluations+=1
            if float((weights*trial)@(weights*trial)) < cost-1e-16:
                q=candidate;accepted=True;break
        if not accepted:reason='local_stagnation';break
    final=pose_error(target,chain.fk(q,tool_offset));evaluations+=1
    if not np.isfinite(q).all() or not np.isfinite(final).all() or np.any(q<lower) or np.any(q>upper):
        raise ValueError('Invalid final IK state')
    if np.linalg.norm(final[:3])<=1e-4 and np.linalg.norm(final[3:])<=1e-3:
        success=True;reason='converged'
    return {'success':success,'reason':reason,'q':q.tolist(),'position_error_m':float(np.linalg.norm(final[:3])),
        'rotation_error_rad':float(np.linalg.norm(final[3:])),'iterations':len(history),'fk_evaluations':evaluations,
        'local_lower':lower.tolist(),'local_upper':upper.tolist(),'trace':history,
        'scope':'Failure means this local solver/branch did not converge; not proof of global unreachability. No collision check.'}
