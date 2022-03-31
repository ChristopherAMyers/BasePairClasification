from openmm.app import PDBFile
from openmm.unit import *
import numpy as np
import sys

#   maximum hydrogen - acceptor distance in angstroms
HBOND_DIST = 2.8

norm_pos = []

class pairInfo():
    def __init__(self) -> None:
        self.angle12 = None
        self.angle1 = None
        self.angle2 = None
        self.com_dist = None
        self.closest_heavy_dist = None
        self.pair_type = None
        self.closest_dist = None


def get_best_fit_plane(pos, center=None):
    #   singular value decomp. to get best the vectors of a plane
    #   that best fits to the base atoms
    pos /= angstroms
    if center is None:
        center = np.mean(pos, axis=0)
    else:
        center /= angstroms
    pos_centered = pos - center
    u, s, vt = np.linalg.svd(pos_centered, full_matrices=False)
    
    norm_pos.append(center)
    norm_pos.append(vt[2] + center)

    return vt[2]


def get_center_of_mass(atoms, pos):
    masses = np.array([atom.element.mass/dalton for atom in atoms])
    com = np.sum(pos*masses[:, None], axis=0)/np.sum(masses)
    return com

def get_pair_type(atoms1, atoms2, positions):
    positions1 = positions[[atom.index for atom in atoms1]]
    positions2 = positions[[atom.index for atom in atoms2]]
    com1 = get_center_of_mass(atoms1, positions1)
    com2 = get_center_of_mass(atoms2, positions2)

    com_diff = com1/angstroms - com2/angstroms
    com_dist = np.linalg.norm(com_diff)

    #   return anything 10 angstroms or greater between COM
    result = pairInfo()
    if com_dist > 10:
        return result

    #   find hydrogen bonds that exist
    hbonds = {}
    for pos1, atom1 in zip(positions1, atoms1):
        for pos2, atom2 in zip(positions2, atoms2):
            #   determine hydrogen and h-aceptor atoms
            hydro = heavy = None
            if atom1.element.symbol == 'H' and atom2.element.symbol in ['N', 'O']:
                hydro = atom1
                heavy = atom2
            elif atom2.element.symbol == 'H' and atom1.element.symbol in ['N', 'O']:
                hydro = atom2
                heavy = atom1


            #   if this is a proper h-bond pair, use closest distance as atom pair
            if hydro is not None and heavy is not None:
                dist = np.linalg.norm(pos1/angstroms - pos2/angstroms)
                
                if dist < HBOND_DIST:
                    #   new pair, just add to dict
                    if hydro not in hbonds: 
                        hbonds[hydro] = (heavy, dist)
                    # update dict if we found a closer heavy atom
                    else:
                        if dist < hbonds[hydro][1]:
                            hbonds[hydro] = (heavy, dist)

    #   caluclate center of mass distance, closest distance between heavy atoms, and overall closest distance
    dCom = com_diff / np.linalg.norm(com_diff)
    heavy1 = np.array([x for x, atom in zip(positions1/angstroms, atoms1) if atom.element.symbol != 'H'])*angstroms
    heavy2 = np.array([x for x, atom in zip(positions2/angstroms, atoms2) if atom.element.symbol != 'H'])*angstroms
    closest_heavy_heavy = np.min(np.linalg.norm(heavy1 - heavy2[:, None], axis=-1))
    closest_dist = np.min(np.linalg.norm(positions1/angstroms - (positions2/angstroms)[:, None], axis=-1))

    #   calculate angles between COM vectors and the vector that is normal to the base plane
    plane1 = get_best_fit_plane(heavy1)
    plane2 = get_best_fit_plane(heavy2)
    angle12 = np.arccos(np.dot(plane1, plane2))*180/np.pi
    if angle12 > 90:
        plane2 = -plane2
        angle12 = np.arccos(np.dot(plane1, plane2))*180/np.pi
    angle1 = np.arccos(np.dot(dCom, plane1))*180/np.pi
    angle2 = np.arccos(np.dot(dCom, plane2))*180/np.pi

    #   attempt to clasify types of base pairs
    pair_type = 'UNK'
    if len(hbonds) > 0 and (angle1 > 60 and angle1 < 120):
        pair_type = 'HBOND'
    else:
        if closest_heavy_heavy > 5.0:
            return result
        if (angle1 < 35 or angle1 > 145) and com_dist < 4.5:
            pair_type = 'STACK'
        elif ((angle1 > 30 and angle1 < 60) or (angle1 > 120 and angle1 < 150)):
            pair_type = 'STAIR'
        else:
            pair_type = 'UNK'
    
    result.pair_type = pair_type
    result.angle12 = angle12
    result.angle1 = angle1
    result.angle2 = angle2
    result.com_dist = com_dist
    result.closest_heavy_dist = closest_heavy_heavy
    result.closest_dist = closest_dist
    return result


def rename_hydrogens(topology):
    #   rename hydrogen atoms using their appropriate host atom names.
    #   this is useful if PyMOL adds hydrogens with generic names
    atoms = list(topology.atoms())

    #   determine number of hydrogens bonded to each heavy atom
    bonded_to_hydros = [set() for atom in atoms]
    for bond in topology.bonds():

        if bond.atom1.element.symbol == 'H':
            bonded_to_hydros[bond.atom2.index].add(bond.atom1.index)
        elif bond.atom2.element.symbol == 'H':
            bonded_to_hydros[bond.atom1.index].add(bond.atom2.index)

    #   rename hydrogens based on number of hydrogens
    for heavy in atoms:
        hydros = [atoms[x] for x in bonded_to_hydros[heavy.index]]
        for n, H in enumerate(hydros):
            new_name = 'H' + heavy.name[1:]
            if len(hydros) > 1:
                new_name += str(n + 1)
            H.name = new_name

def get_clasifications(pdb_file):
    pdb = PDBFile(pdb_file)
    positions = pdb.getPositions(True)
    rename_hydrogens(pdb.topology)

    results = {}
    for res1 in pdb.topology.residues():
        for res2 in pdb.topology.residues():

            #   ignore identical and modified residues and 
            if res2.index <= res1.index: continue
            if res1.name not in ['A', 'C', 'G', 'U']: continue
            if res2.name not in ['A', 'C', 'G', 'U']: continue

            atoms1 = []
            atoms2 = []
            for atom in res1.atoms():
                #   ignore sugar and phosphate atoms
                if atom.name[-1] == "'": continue
                if atom.name in ['P', 'OP1', 'OP2']: continue
                atoms1.append(atom)
            for atom in res2.atoms():
                #   ignore sugar and phosphate atoms
                if atom.name[-1] == "'": continue
                if atom.name in ['P', 'OP1', 'OP2']: continue
                atoms2.append(atom)

            pair_info = get_pair_type(atoms1, atoms2, positions)
            if pair_info.pair_type: 
                if pair_info.pair_type not in results:
                    results[pair_info.pair_type] = []
                results[pair_info.pair_type].append(pair_info)

    for pair_type, pair_data in results.items():
        for pair in pair_data:
            print("{:6s}:  {:3s} {:3s} - {:3s} {:3s}  {:6.1f} {:6.1f} {:6.1f} | {:8.2f}  {:8.2f}  {:8.2f}".format(pair_type, res1.name, res1.id, res2.name, res2.id, pair.angle12, pair.angle1, pair.angle2, pair.com_dist, pair.closest_heavy_dist, pair.closest_dist))
        print()

    #   debug only: print out vector normals to each base from SVD
    if False:
        with open('tmp.xyz', 'w') as file:
            file.write('{:d}\n'.format(len(norm_pos)))
            file.write("norm pos \n")
            for x, y, z in norm_pos:
                file.write('He {:10.5f}  {:10.5f}  {:10.5f} \n'.format(x, y, z))

    return results

if __name__ == '__main__':
    get_clasifications(sys.argv[1])
