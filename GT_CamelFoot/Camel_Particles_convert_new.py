"""Create an overlapping, HOLLOW camel-foot particle skin directly from an STL.

Run: pip install numpy scipy trimesh

Place this script, 'Part Studio 1.stl' and  settled soil .data file
in the same folder. Makes camel_foot_surface.txt (only outer shell) and
camel_foot_surface_on_soil.data (original soil + camel-foot shell).
"""

from pathlib import Path
from itertools import product
import re
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
STL = ROOT / 'Part Studio 1.stl'
SOIL_FILES = [
    ROOT / 'soil_settled_p4_150_100_150_ppr10_12345(2).data',
    ROOT / 'soil_settled_p4_150_100_150_ppr10_12345.data',
]
SOIL = next((p for p in SOIL_FILES if p.is_file()), None)
XYZ_OUT = ROOT / 'camel_foot_surface_new.txt'
DATA_OUT = ROOT / 'camel_foot_surface_on_soil_new.data'

# Measured in meters. The uploaded STL already uses meters.
PARTICLE_DIAMETER = 0.0020
OVERLAP_FRACTION = 0.35   # 35% of diameter overlap between nearby foot spheres
SPACING = PARTICLE_DIAMETER * (1 - OVERLAP_FRACTION)
SOIL_CLEARANCE = 0.0002  # gap between underside of foot spheres and soil top
DENSITY = 2670
SOIL_TYPE = 1
SEED = 12345

# -------- 1) Surface sampling, not volume filling --------
mesh = trimesh.load_mesh(STL, process=True)
if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
    raise ValueError('STL must contain a nonempty triangle mesh')
if not mesh.is_watertight:
    print('Warning: STL is not closed; only the existing triangles will be sampled.')

rng = np.random.default_rng(SEED)
triangles = mesh.triangles
probabilities = mesh.area_faces / mesh.area

# Spatial-hash Poisson-disc sampling: all accepted centers lie ON the STL
# triangles, and no pair is closer than SPACING. Their spheres overlap
# because PARTICLE_DIAMETER > SPACING.
cells = {}
accepted = []
neighbor_offsets = list(product((-1, 0, 1), repeat=3))

def add_if_far_enough(p):
    key = tuple(np.floor(p / SPACING).astype(int))
    #for offset in neighbor_offsets:
        #neighbors = cells.get(tuple(key[a] + offset[a] for a in range(3)))
        #if neighbors:
            #for j in neighbors:
                # ***CHANGE THIS!!!!!***
                #if np.dot(accepted[j] - p, accepted[j] - p) < SPACING**2:
                    #return
    j = len(accepted)
    accepted.append(np.array(p, dtype=float))
    cells.setdefault(key, []).append(j)

# Start with corners and triangle edges, so the outline stays well defined.
for point in mesh.vertices:
    add_if_far_enough(point)
for i, j in mesh.edges_unique:
    a, b = mesh.vertices[i], mesh.vertices[j]
    steps = max(1, int(np.ceil(np.linalg.norm(b - a) / SPACING)))
    for t in np.linspace(0, 1, steps + 1):
        add_if_far_enough((1 - t) * a + t * b)

# Evenly distributed random candidate points ON triangles. Sampling at
# each iteration fills remaining gaps without adding any interior points.
rough_count = max(1000, int(np.ceil(mesh.area / SPACING**2)))
for iteration in range(10):
    face_indices = rng.choice(len(triangles), size=3 * rough_count, p=probabilities)
    tri = triangles[face_indices]
    u = np.sqrt(rng.random(len(tri)))
    v = rng.random(len(tri))
    points = ((1 - u)[:, None] * tri[:, 0] +
              (u * (1 - v))[:, None] * tri[:, 1] +
              (u * v)[:, None] * tri[:, 2])
    old_n = len(accepted)
    for point in points:
        add_if_far_enough(point)
    print(f'Surface pass {iteration + 1}: {len(accepted)} particles')
    if len(accepted) - old_n < 0.002 * len(accepted):
        break

foot_xyz = np.array(accepted)
np.savetxt(XYZ_OUT, foot_xyz, fmt='%.9f', header='x y z', comments='')
print(f'Outer shell saved to: {XYZ_OUT.name}')

# -------- 2) Rotate Z-up foot so LAMMPS Y is up --------
foot = foot_xyz[:, [0, 2, 1]].copy()
foot[:, 2] *= -1

# -------- 3) Parse the original sphere-style LAMMPS soil file --------
if SOIL is None:
    raise FileNotFoundError('Put the original settled soil .data file next to the script.')
lines = SOIL.read_text().splitlines(keepends=True)

SECTION_HEADERS = {
    'Masses', 'Atoms', 'Velocities', 'Bonds', 'Angles', 'Dihedrals',
    'Impropers', 'Pair Coeffs', 'PairIJ Coeffs', 'Bond Coeffs',
    'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs',
    'Ellipsoids', 'Lines', 'Triangles', 'Bodies', 'Molecules',
}

def section_start(name):
    return next((i for i, ln in enumerate(lines)
                 if ln.split('#', 1)[0].strip() == name), None)

def section_end(start):
    return next((i for i in range(start + 1, len(lines))
                 if lines[i].split('#', 1)[0].strip() in SECTION_HEADERS), len(lines))

atom_start = section_start('Atoms')
if atom_start is None or 'sphere' not in lines[atom_start].lower():
    raise ValueError('Expected an Atoms # sphere section.')
atom_end = section_end(atom_start)

rows = []
for ln in lines[atom_start + 1:atom_end]:
    fields = ln.split('#', 1)[0].split()
    if fields and fields[0].isdigit():
        if len(fields) < 7:
            raise ValueError(f'Invalid sphere-atom line: {ln}')
        rows.append([float(fields[i]) for i in (0, 1, 2, 4, 5, 6)])
soil = np.asarray(rows)  # ID, type, diameter, x, y, z
if not len(soil):
    raise ValueError('Soil file contains no sphere atoms.')
if len(np.unique(soil[:, 0])) != len(soil):
    raise ValueError('Input soil file contains duplicate atom IDs.')
bed = soil[soil[:, 1] == SOIL_TYPE]
if not len(bed):
    raise ValueError(f'No particles with SOIL_TYPE={SOIL_TYPE}.')
next_id = int(soil[:, 0].max()) + 1
old_types = next((int(m.group(1)) for ln in lines[:atom_start]
                  if (m := re.fullmatch(r'\s*(\d+)\s+atom types\s*', ln))),
                 int(soil[:, 1].max()))
foot_type = max(old_types, int(soil[:, 1].max())) + 1

# Center above the soil in the horizontal X,Z directions.
for foot_axis, soil_col in ((0, 3), (2, 5)):
    center_foot = (foot[:, foot_axis].min() + foot[:, foot_axis].max()) / 2
    center_soil = (bed[:, soil_col].min() + bed[:, soil_col].max()) / 2
    foot[:, foot_axis] += center_soil - center_foot

# Soil top measured under the shape footprint (including sphere radii).
radius = PARTICLE_DIAMETER / 2
margin = bed[:, 2].max() / 2 + radius
mask = ((bed[:, 3] >= foot[:, 0].min() - margin) &
        (bed[:, 3] <= foot[:, 0].max() + margin) &
        (bed[:, 5] >= foot[:, 2].min() - margin) &
        (bed[:, 5] <= foot[:, 2].max() + margin))
if not np.any(mask):
    raise ValueError('Foot footprint does not overlap the soil bed.')
soil_top = np.max(bed[mask, 4] + bed[mask, 2] / 2)
foot[:, 1] += soil_top + SOIL_CLEARANCE + radius - foot[:, 1].min()

# -------- 4) Update counts, type count and boundaries --------
new_total = len(soil) + len(foot)
seen_count = seen_types = False
for i in range(atom_start):
    ln = lines[i]
    if re.fullmatch(r'\s*\d+\s+atoms\s*', ln):
        lines[i] = f'{new_total} atoms\n'
        seen_count = True
    elif re.fullmatch(r'\s*\d+\s+atom types\s*', ln):
        lines[i] = f'{foot_type} atom types\n'
        seen_types = True
    else:
        for axis, (lo, hi) in enumerate((('xlo', 'xhi'), ('ylo', 'yhi'), ('zlo', 'zhi'))):
            m = re.fullmatch(rf'\s*(\S+)\s+(\S+)\s+{lo}\s+{hi}\s*', ln)
            if m:
                lower = min(float(m.group(1)), foot[:, axis].min() - radius)
                upper = max(float(m.group(2)), foot[:, axis].max() + radius)
                lines[i] = f'{lower:.12g} {upper:.12g} {lo} {hi}\n'
                break
if not (seen_count and seen_types):
    raise ValueError('Missing atom count or atom types in soil file header.')

# -------- 5) Append the shell particles without changing soil rows --------
new_atoms = [
    f'{next_id+i} {foot_type} {PARTICLE_DIAMETER:.12g} {DENSITY:.12g} '
    f'{x:.12g} {y:.12g} {z:.12g} 0 0 0\n'
    for i, (x, y, z) in enumerate(foot)
]
insert_at = section_end(atom_start)
while insert_at > atom_start + 1 and not lines[insert_at - 1].strip():
    insert_at -= 1
if insert_at > 0 and not lines[insert_at - 1].endswith('\n'):
    lines[insert_at - 1] += '\n'
lines[insert_at:insert_at] = new_atoms

velocity_start = section_start('Velocities')
if velocity_start is not None:
    vel_end = section_end(velocity_start)
    while vel_end > velocity_start + 1 and not lines[vel_end - 1].strip():
        vel_end -= 1
    lines[vel_end:vel_end] = [f'{next_id+i} 0 0 0 0 0 0\n' for i in range(len(foot))]

DATA_OUT.write_text(''.join(lines))

# Confirm surface point distribution (nearest-neighbor center distances).
tree = cKDTree(foot_xyz)
near = tree.query(foot_xyz, k=2)[0][:, 1]
print(f'Surface particles: {len(foot_xyz)}')
print(f'Foot particle diameter: {PARTICLE_DIAMETER:g} m')
print(f'Nearest-center distances: {near.min():.6g} to {near.max():.6g} m')
print(f'Foot-to-soil clearance: {SOIL_CLEARANCE:g} m')
print(f'Original soil atoms: {len(soil)}; total atoms: {new_total}')
print(f'Camel foot atom type: {foot_type}')
print(f'Combined LAMMPS file saved to: {DATA_OUT.name}')
