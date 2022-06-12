import numpy as np
import os
import scipy.spatial
import multiprocessing
import concurrent.futures

from Bio import PDB
from pathlib import Path

from Bio.PDB import calc_dihedral, calc_angle
from argparse import ArgumentParser
from RRCS.RRCS import calc_contact
import random

from utils_training import parse_a3m
from collections import defaultdict

from concurrent.futures import ProcessPoolExecutor

NCPU = 4
retain_all_res = True

res_name_dict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'PHD': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
                 'GLY': 'G',
                 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'MSE': 'M', 'PHE': 'F', 'PRO': 'P',
                 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


# https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
def get_dihedrals(a, b, c, d):
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:, None]

    v = b0 - np.sum(b0 * b1, axis=-1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=-1)[:, None] * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)

    return np.arctan2(y, x)


def get_angles(a, b, c):
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:, None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:, None]

    x = np.sum(v * w, axis=1)

    return np.arccos(x)


def get_neighbors(residues, dmax):
    nres = len(residues)

    # three anchor atoms
    N = np.stack([np.array(residues[i]['N'].coord) for i in range(nres)])
    Ca = np.stack([np.array(residues[i]['CA'].coord) for i in range(nres)])
    C = np.stack([np.array(residues[i]['C'].coord) for i in range(nres)])

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

    for i in range(nres):
        resname = residues[i].resname
        resname = resname.strip()
        if (resname != "GLY"):
            Cb[i] = np.array(residues[i]['CB'].coord)

    # fast neighbors search
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    # indices of contacting residues
    idx = np.array([[i, j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist6d = np.zeros((nres, nres))
    dist6d[idx0, idx1] = np.linalg.norm(Cb[idx1] - Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres))
    omega6d[idx0, idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres))
    theta6d[idx0, idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres))
    phi6d[idx0, idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    return dist6d, omega6d, theta6d, phi6d


def parse_pdb_6d(pid, a3m_file, pdb_file, save_pth):
    # load PDB
    pp = PDB.PDBParser()
    structure = pp.get_structure('myStructureName', pdb_file)[0]
    chain = structure.child_list[0]
    if retain_all_res:
        residues = [res for res in chain.child_list if PDB.is_aa(res)]
    else:
        residues = [res for res in chain.child_list if PDB.is_aa(res) and 'CB' in res and 'N' in res]

    msa = parse_a3m(a3m_file, limit=30000)

    if not msa.shape[1]==len(residues):
        return

    if len(residues) > 260:
        os.makedirs(save_pth, exist_ok=True)
        with open(f'{save_pth}/long_list_all', 'a') as f:
            f.write(f'{pid}_long\n')
    
    #calculate RRCS
    contact = calc_contact(pdb_file)
    rrcs = []
    for a_res in contact:
        b_res_list = contact[a_res].keys()
        for b_res in b_res_list:
            score = contact[a_res][b_res]
            if score > 0:
                rrcs.append(score)
    rrcs = random.sample(rrcs, len(residues))
    if len(rrcs)<len(residues):
        for i in range(len(residues)-len(rrcs)): 
            rrcs.append(0) 
    rrcs = np.array(rrcs)

    # 6D coordinates
    dist, omega, theta_asym, phi_asym = get_neighbors(residues, 20)

    labels = {
        'msa': msa, 'rrcs':rrcs,
        'dist': dist, 'omega': omega,
        'theta_asym': theta_asym, 'phi_asym': phi_asym
    }

    print()
    print('pid:',pid)
    print('msa:',msa.shape)
    print('rrcs:',rrcs.shape)
    print('dist:',dist.shape)
    print('omega:',omega.shape)
    print('theta_asym:',theta_asym.shape)
    print('phi_asym:',phi_asym.shape)
    print('residues:',len(residues))

    print()


    np.savez_compressed(f'{save_pth}/{pid}.npy', **labels)


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument('-a3m', '--a3m_pth', type=str, default='/home/yangjianyi_mem/project/trrosetta/training_set/a3m/', help='path to A3M files of training set')
    # parser.add_argument('-pdb', '--pdb_pth', type=str, default='/home/yangjianyi_mem/project/trrosetta/training_set/pdb/', help='path to PDB files of training set')
    # parser.add_argument('-o', '--out_pth', type=str, default='/home/yangjianyi_mem/wangwk/temp/npz/', help='path to store npz files')
    parser.add_argument('-a3m', '--a3m_pth', type=str, required=True, help='path to A3M files of training set')
    parser.add_argument('-pdb', '--pdb_pth', type=str, required=True, help='path to PDB files of training set')
    parser.add_argument('-o', '--out_pth', type=str, required=True, help='path to store npz files')
    parser.add_argument('-cpu', '--n_cpu', type=int, default=2, help='num of CPU cores to use')
    args = parser.parse_args()

    pid_lst = [f.split('.')[0] for f in os.listdir(args.a3m_pth)]
    # print(pid_lst)

    executor = concurrent.futures.ProcessPoolExecutor(args.n_cpu)
    futures = [executor.submit(parse_pdb_6d, pid, f'{args.a3m_pth}/{pid}.a3m', f'{args.pdb_pth}/{pid}.pdb', args.out_pth) for pid in pid_lst]
    results = concurrent.futures.wait(futures)
