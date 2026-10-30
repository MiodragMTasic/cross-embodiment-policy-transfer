"""Column-vector FK/Jacobian from exported USD joint frames; no simulator imports."""
import numpy as np


def rotation_wxyz(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-4:
        raise ValueError('Expected finite unit wxyz quaternion')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def pose_matrix(position,quaternion):
    t=np.eye(4);t[:3,:3]=rotation_wxyz(quaternion);t[:3,3]=position
    if not np.isfinite(t).all():raise ValueError('Nonfinite pose')
    return t


def rigid_inverse(t):
    t=np.asarray(t,dtype=float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1],atol=1e-9):
        raise ValueError('Invalid homogeneous matrix')
    if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-7):
        raise ValueError('Expected proper rigid rotation')
    inverse=np.eye(4);inverse[:3,:3]=t[:3,:3].T;inverse[:3,3]=-t[:3,:3].T@t[:3,3]
    return inverse


def axis_rotation(axis,angle):
    k=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0.]])
    t=np.eye(4);t[:3,:3]=np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)
    return t


class RigidChain:
    def __init__(self,graph):
        self.names=list(graph['arm_joint_names'])
        if len(self.names)!=len(set(self.names)):raise ValueError('Repeated joint name')
        self.edges=[]
        cursor=graph['base_body']
        active=[]
        for item in graph['chain']:
            j=graph['joints'][item['joint_index']];direction=item['direction']
            if direction not in (-1,1) or not j['enabled']:raise ValueError('Invalid edge')
            start,end=(j['body0'],j['body1']) if direction==1 else (j['body1'],j['body0'])
            if start!=cursor:raise ValueError('Disconnected joint chain')
            cursor=end
            t0=pose_matrix(j['local_pos0'],j['local_rot0_wxyz'])
            t1=pose_matrix(j['local_pos1'],j['local_rot1_wxyz'])
            if j['type']=='PhysicsRevoluteJoint':
                idx=self.names.index(j['name']);active.append(j['name'])
                axis=np.eye(3)['XYZ'.index(j['axis'])]
            elif j['type']=='PhysicsFixedJoint':idx=None;axis=None
            else:raise ValueError('Unsupported joint type')
            self.edges.append((t0,t1,direction,idx,axis))
        if cursor!=graph['raw_tool_body'] or active!=self.names:raise ValueError('Wrong tool or joint ordering')

    def fk_jacobian(self,q,tool_offset=None):
        q=np.asarray(q,dtype=float)
        if q.shape!=(len(self.names),) or not np.isfinite(q).all():raise ValueError('Wrong joint vector')
        t=np.eye(4);axes={}
        for t0,t1,direction,idx,axis in self.edges:
            start,finish=(t0,t1) if direction==1 else (t1,t0)
            before=t@start
            if idx is None:t=before@rigid_inverse(finish)
            else:
                axes[idx]=(before[:3,3].copy(),direction*(before[:3,:3]@axis))
                t=before@axis_rotation(axis,direction*q[idx])@rigid_inverse(finish)
        if tool_offset is not None:
            rigid_inverse(tool_offset)  # validate before applying
            t=t@tool_offset
        jac=np.zeros((6,len(self.names)))
        for idx,(origin,axis) in axes.items():
            jac[:3,idx]=np.cross(axis,t[:3,3]-origin);jac[3:,idx]=axis
        return t,jac

    def fk(self,q,tool_offset=None):
        return self.fk_jacobian(q,tool_offset)[0]
