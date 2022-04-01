from openmm.app import PDBFile
from openmm.unit import *
import numpy as np
from math import cos, sin
import sys
from copy import deepcopy

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
        self.atoms1 = []
        self.atoms2 = []
        self.positions1 = []
        self.positions2 = []
        self.projected_area = 0


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

def get_stack_angle_sum(heavy_pos1, heavy_pos2):

    dists = np.linalg.norm(heavy_pos1 - heavy_pos2[:, None])
    
def get_stacked_projection_area(heavy1, heavy2, norm1, norm2, c1, c2):

    c1 /= angstroms
    c2 /= angstroms
    heavy1 /= angstroms
    heavy2 /= angstroms

    hit_points = []
    all_points = []

    #   re-center based on c1
    atoms1 = heavy1 - c1
    atoms2 = heavy2 - c1
    c2 = (c2 - c1)

    #   re-orient so that norm1 points along the z-axis
    norm1 = norm1/np.linalg.norm(norm1)
    theta = -np.arccos(norm1[2])
    phi   = -np.arctan2(norm1[1], norm1[0])
    rot_z = np.array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
    rot_y = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
    rot_zy = rot_y @ rot_z
    atoms1 = np.dot(rot_zy, atoms1.T).T
    atoms2 = np.dot(rot_zy, atoms2.T).T
    norm1 = rot_zy @ np.copy(norm1)
    norm2 = rot_zy @ np.copy(norm2)
    c2 = rot_zy @ c2

    #   rays are directed from plane one in the direction of norm 1 and the
    #   intersection with plane 2 is calculated. If the intersection 
    area_sum = 0
    intersect_area_sum = 0
    dx = 0.2
    max_extent_1 = np.max(np.linalg.norm(atoms1, axis=1)) + 1.5
    max_extent_2 = np.max(np.linalg.norm(atoms2 - c2, axis=1)) + 1.5
    search_points = np.arange(-max_extent_1, max_extent_1, dx)
    for x in search_points:
        for y in search_points:
            p1 = np.array([x, y, 0])
            
            if np.min(np.linalg.norm(p1 - atoms1, axis=1)) > 1.5: continue
            area_sum += 1
            all_points.append(p1)
            intersect_param = np.dot(c2 - p1, norm2)/np.dot(norm1, norm2)
            intersect_point = p1 + intersect_param*norm1
            
            if np.min(np.linalg.norm(intersect_point - atoms2, axis=1)) < 1.5:
                intersect_area_sum += 1
                hit_points.append(intersect_point)
            
    
    max_points = len(search_points)**2
    plane1_area = (area_sum/max_points)
    intersection_area = (intersect_area_sum/max_points)

    if True:
        with open('tmp.xyz', 'w') as file:
            file.write('{:d} \n'.format(len(atoms1) + len(atoms2) + len(hit_points) + len(all_points)))
            file.write('debug \n')
            for coord in atoms1:
                file.write('C {:10.5f}  {:10.5f}  {:10.5f} \n'.format(*tuple(coord)))
            for coord in atoms2:
                file.write('C {:10.5f}  {:10.5f}  {:10.5f} \n'.format(*tuple(coord)))
            for coord in hit_points:
                file.write('He {:10.5f}  {:10.5f}  {:10.5f} \n'.format(*tuple(coord)))
            for coord in all_points:
                file.write('Ne {:10.5f}  {:10.5f}  {:10.5f} \n'.format(*tuple(coord)))

    return intersection_area/plane1_area


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

    area_ratio_1 = get_stacked_projection_area(heavy1, heavy2, plane1, plane2, com1, com2)
    area_ratio_2 = get_stacked_projection_area(heavy2, heavy1, plane2, plane1, com2, com1)
    area_ratio = np.max((area_ratio_1, area_ratio_2))
    result.projected_area = area_ratio



    #   attempt to clasify types of base pairs
    pair_type = 'UNK'
    if len(hbonds) > 0 and (angle1 > 60 and angle1 < 120):
        pair_type = 'HBOND'
    else:
        if closest_heavy_heavy > 5.0:
            return result

        if (angle12 < 45 or angle12 >  135) and area_ratio > 0.20:
            pair_type = 'STACK'
        elif (angle12 < 45 or angle12 >  135)  and area_ratio < 0.20:
            pair_type = 'STAIR'
        # if (angle1 < 35 or angle1 > 145) and com_dist < 4.5:
        #     pair_type = 'STACK'
        # elif ((angle1 > 30 and angle1 < 60) or (angle1 > 120 and angle1 < 150)):
        #     pair_type = 'STAIR'
        else:
            pair_type = 'UNK'
    
    result.pair_type = pair_type
    result.angle12 = angle12
    result.angle1 = angle1
    result.angle2 = angle2
    result.com_dist = com_dist
    result.closest_heavy_dist = closest_heavy_heavy
    result.closest_dist = closest_dist
    result.atoms1 = atoms1
    result.atoms2 = atoms2
    result.positions1 = deepcopy(positions1)
    result.positions2 = deepcopy(positions2)
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

def get_clasifications(pdb_file, print_results=False):
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
                if "'" in atom.name: continue
                if atom.name in ['P', 'OP1', 'OP2', 'OP3', 'OP4', 'HP3', 'HP4']: continue
                atoms1.append(atom)
            for atom in res2.atoms():
                #   ignore sugar and phosphate atoms
                if  "'" in atom.name: continue
                if atom.name in ['P', 'OP1', 'OP2', 'OP3', 'OP4', 'HP3', 'HP4']: continue
                atoms2.append(atom)

            pair_info = get_pair_type(atoms1, atoms2, positions)
            if pair_info.pair_type: 
                if pair_info.pair_type not in results:
                    results[pair_info.pair_type] = []
                results[pair_info.pair_type].append(pair_info)

    if print_results:
        for pair_type, pair_data in results.items():
            print("{:6s}   {:3s} {:3s} - {:3s} {:3s}  {:>12s} | {:>12s}  {:>12s}  {:>12s} | {:>12}" \
                    .format("", "Res", "ID", "Res", "ID", "Angle-norms",  \
                        "COM-dist", "Closest dist", "Closest dist", "Proj. Area"))
            print("{:6s}   {:3s} {:3s} - {:3s} {:3s}  {:>12s} | {:>12s}  {:>12s}  {:>12s} | {:>12s}" \
                    .format("", "", "", "", "", "",  \
                        "", "Heavy-Heavy", "All-All", ""))
            for pair in pair_data:
                res1 = pair.atoms1[0].residue
                res2 = pair.atoms2[0].residue
                print("{:6s}:  {:3s} {:3s} - {:3s} {:3s}  {:12.1f} | {:12.2f}  {:12.2f}  {:12.2f} | {:12.5f}" \
                    .format(pair_type, res1.name, res1.id, res2.name, res2.id, pair.angle12, \
                        pair.com_dist, pair.closest_heavy_dist, pair.closest_dist, pair.projected_area))
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
    get_clasifications(sys.argv[1], print_results=True)
