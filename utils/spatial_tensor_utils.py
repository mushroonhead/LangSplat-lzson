"""
Spatial manipulation utils implemented with vectorized tensor operations
"""

import torch
from typing import Iterable, Tuple

def quat2rot(quat: torch.Tensor) -> torch.Tensor:
    """
    From: https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    - Inputs:
        - quat: (...,4) tensor, rxyz
    - Returns:
        - rot: (...,3,3) tensor 
    """
    rot = torch.stack(
        (torch.stack((1 - 2*(quat[...,2]*quat[...,2] + quat[...,3]*quat[...,3]), 
                      2*(quat[...,1]*quat[...,2] - quat[...,0]*quat[...,3]), 
                      2*(quat[...,1]*quat[...,3] + quat[...,0]*quat[...,2])), dim=-1),
         torch.stack((2*(quat[...,1]*quat[...,2] + quat[...,0]*quat[...,3]), 
                      1 - 2*(quat[...,1]*quat[...,1] + quat[...,3]*quat[...,3]), 
                      2*(quat[...,2]*quat[...,3] - quat[...,0]*quat[...,1])), dim=-1),
         torch.stack((2*(quat[...,1]*quat[...,3] - quat[...,0]*quat[...,2]), 
                      2*(quat[...,2]*quat[...,3] + quat[...,0]*quat[...,1]), 
                      1 - 2*(quat[...,1]*quat[...,1] + quat[...,2]*quat[...,2])), dim=-1)), dim=-2)
    return rot

def rot2quat(rot: torch.Tensor) -> torch.Tensor:
    """
    Vectorized for 1 dim only (code requirements complex)
    From: https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    - Inputs:
        - rot: (B,3,3) tensor
    - Returns:
        - quat: (B,4) tensor, rxyz
    """
    # find magnitude first
    mag = torch.stack(
        (1 + rot[:,0,0] + rot[:,1,1] + rot[:,2,2],
         1 + rot[:,0,0] - rot[:,1,1] - rot[:,2,2],
         1 - rot[:,0,0] + rot[:,1,1] - rot[:,2,2],
         1 - rot[:,0,0] - rot[:,1,1] + rot[:,2,2],), dim=-1).sqrt()/2 # (B,4)
    # find case and do appropriate changes
    case_num = mag.argmax(dim=-1) #(B,)
    eye = torch.eye(rot.shape[0], dtype=rot.dtype, device=rot.device)

    c0 = (case_num == 0) # filter case 0
    h0 = eye[c0].transpose(-1,-2).to_sparse() # recover shape
    m0, f0 = mag[c0], rot[c0] # filtered magnitude and quaternions
    c0_quats = torch.stack(
        (m0[:,0],
         (f0[:,2,1] - f0[:,1,2])/(4*m0[:,0]),
         (f0[:,0,2] - f0[:,2,0])/(4*m0[:,0]),
         (f0[:,1,0] - f0[:,0,1])/(4*m0[:,0])), dim=-1)

    c1 = (case_num == 1) # filter case 1
    h1 = eye[c1].transpose(-1,-2).to_sparse() # recover shape
    m1, f1 = mag[c1], rot[c1] # filtered magnitude and quaternions
    c1_quats = torch.stack(
        ((f1[:,2,1] - f1[:,1,2])/(4*m1[:,1]),
         m1[:,1],
         (f1[:,0,1] + f1[:,1,0])/(4*m1[:,1]),
         (f1[:,0,2] + f1[:,2,0])/(4*m1[:,1])), dim=-1)
    
    c2 = (case_num == 2) # filter case 2
    h2 = eye[c2].transpose(-1,-2).to_sparse() # recover shape
    m2, f2 = mag[c2], rot[c2] # filtered magnitude and quaternions
    c2_quats = torch.stack(
        ((f2[:,0,2] - f2[:,2,0])/(4*m2[:,2]),
         (f2[:,0,1] + f2[:,1,0])/(4*m2[:,2]),
         m2[:,2],
         (f2[:,1,2] + f2[:,2,1])/(4*m2[:,2])), dim=-1)
    
    c3 = (case_num == 3) # filter case 3
    h3 = eye[c3].transpose(-1,-2).to_sparse() # recover shape
    m3, f3 = mag[c3], rot[c3] # filtered magnitude and quaternions
    c3_quats = torch.stack(
        ((f3[:,1,0] - f3[:,0,1])/(4*m3[:,3]),
         (f3[:,0,2] + f3[:,2,0])/(4*m3[:,3]),
         (f3[:,1,2] + f3[:,2,1])/(4*m3[:,3]),
         m3[:,3]), dim=-1)
    
    # shape rotations back to correct space
    quat = h0 @ c0_quats + h1 @ c1_quats + h2 @ c2_quats + h3 @ c3_quats

    return quat.to_dense()

def quat_mult(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """
    Quat multiplication q0q1 (not commutative)
    From: https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    - Input:
        - q0: (...,4) tensor, rxyz
        - q1: (...,4) tensor, rxyz
    - Returns:
        - q0q1: (...,4) tensor, rxyz
    """
    q0q1 = torch.stack(
        (q0[...,0]*q1[...,0] - q0[...,1]*q1[...,1] - q0[...,2]*q1[...,2] - q0[...,3]*q1[...,3],
         q0[...,0]*q1[...,1] + q0[...,1]*q1[...,0] - q0[...,2]*q1[...,3] + q0[...,3]*q1[...,2],
         q0[...,0]*q1[...,2] + q0[...,1]*q1[...,3] + q0[...,2]*q1[...,0] - q0[...,3]*q1[...,1],
         q0[...,0]*q1[...,3] - q0[...,1]*q1[...,2] + q0[...,2]*q1[...,1] + q0[...,3]*q1[...,0]), dim=-1)
    return q0q1

def quat_inv(quat: torch.Tensor) -> torch.Tensor:
    """
    Quat inversion
    From: https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    - Input:
        - quat: (...,4) tensor, rxyz
    - Returns:
        - quat_inv: (...,4) tensor, rxyz
    """
    mult = torch.tensor((1,-1,-1,-1), dtype= quat.dtype, device=quat.device).diag()
    return (mult @ quat[...,None]).squeeze(-1)

def unit_quat(dtype=torch.float32, device=torch.device('cpu')) -> torch.Tensor:
    """
    Generates a unit quat
    - Inputs:
        - dtype: torch.dtype, dtype of generated tensor
        - device: torch.device, device of generated tensor
    - Returns:
        - unit_quat: (4,) tensor, rxyz (1,0,0,0)
    """
    return torch.tensor((1,0,0,0), dtype=dtype, device=device)

def rand_rot_quat(size: Tuple[int,Iterable[int]], 
                  dtype=torch.float32, device=torch.device('cpu')) -> torch.Tensor:
    """
    Random rotation quaternions
    - Inputs:
        - size: int | iterable of int, number of rand rotational quaternions to use
        - dtype: torch.dtype, dtype of generated tensor
        - device: torch.device, device of generated tensor  
    - Returns:
        - rand_quat: (size,4) tensor, rxyz
    """
    return torch.nn.functional.normalize(torch.randn(size, 4, dtype=dtype, device=device))

def getWorld2View2(R : torch.Tensor, t: torch.Tensor, 
                   translate=torch.zeros(3), scale=1.0) -> torch.Tensor:
    """
    Tensor vectorized version of original getWorld2View2 from lang-splat
    - Inputs:
        - R : (...,3,3) tensor, rotation matrix
        - t: (...,3) tensor, translation vector
        - translate: (3,) tensor, translate? (default=[0,0,0])
        - scale: float, scaling factor (default=1.0)
    - Returns:
        - Rt: (...,3,3) tensor, transpose rotation of cam?
        - C2W: (...,3) tensor, camera to world frame translation
    """
    batch_size = R.shape[0]
    Rt = torch.cat(
        (torch.cat((R.transpose(-1,-2), t[...,None]), dim=-1),
        torch.cat((torch.zeros_like(t[...,None,:]),torch.ones(batch_size, 1, 1, device=R.device)), dim=-1)),
        dim=-2)
    Rt = Rt.double()

    C2W = torch.linalg.solve(Rt, torch.eye(4, device=R.device, dtype=torch.float64)) # higher precision inversion
    cam_center = C2W[...,:3, 3]
    cam_center = (cam_center + translate.to(device=R.device, dtype=torch.float64)) * scale
    C2W[...,:3, 3] = cam_center
    Rt = torch.linalg.solve(C2W, torch.eye(4, device=R.device, dtype=torch.float64))
    return Rt.to(dtype=R.dtype), C2W[...,3,:3].to(dtype=R.dtype) # back to original precision


if __name__ == '__main__':
    """
    Sanity checks
    """

    # random quats
    rand_quats = rand_rot_quat(6, dtype=torch.float64)

    # check inversion operation and quat multiply
    rand_quats_inv = quat_inv(rand_quats)
    assert torch.allclose(unit_quat(dtype=torch.float64), quat_mult(rand_quats, rand_quats_inv)), \
        f'Inversion does not give unit rotation, output: {quat_mult(rand_quats, rand_quats_inv)}'
    assert torch.allclose(unit_quat(dtype=torch.float64), quat_mult(rand_quats_inv, rand_quats)), \
        f'Inversion does not give unit rotation, output: {quat_mult(rand_quats_inv, rand_quats)}'
    
    # check quat2rot using inversion
    rand_rots = quat2rot(rand_quats)
    rand_rots_inv = quat2rot(rand_quats_inv)
    assert torch.allclose(torch.eye(3, dtype=torch.float64), rand_rots @ rand_rots_inv), \
        f'Inversion does not give unit rotation, output: {rand_rots @ rand_rots_inv}'
    assert torch.allclose(torch.eye(3, dtype=torch.float64), rand_rots_inv @ rand_rots), \
        f'Inversion does not give unit rotation, output: {rand_rots_inv @ rand_rots}'
    
    # check rot2quat using inversion
    rand_rots_2 = rot2quat(rand_rots)
    rand_rots_inv_2 = rot2quat(rand_rots_inv)
    # check includes an abs due to 2 possible quat -> quaternion and its negative
    assert torch.allclose(unit_quat(dtype=torch.float64), quat_mult(rand_rots_2, rand_rots_inv_2).abs()), \
        f'Inversion does not give unit rotation, output: {quat_mult(rand_rots_2, rand_rots_inv_2).abs()}'
    assert torch.allclose(unit_quat(dtype=torch.float64), quat_mult(rand_rots_inv_2, rand_rots_2).abs()), \
        f'Inversion does not give unit rotation, output: {quat_mult(rand_rots_inv_2, rand_rots_2).abs()}'