from dolfin import mesh
from spuq.fem.fenics.fenics_utils import create_joint_mesh

import pickle
import os
import glob


def generate_reference_setup(basedir, SOLUTIONFN='SFEM2-SOLUTIONS-P?.pkl'):
    # ============================================================
    # import solutions and determine reference setting
    # ============================================================
    for i, fn in enumerate(glob.glob(os.path.join(basedir, SOLUTIONFN))):
        print "--------> loading data from", fn
        with open(fn, 'rb') as fin:
            w_history = pickle.load(fin)
            mesh0 = w_history[-1].basis.basis.mesh
            Lambda0 = w_history[-1].active_indices()
            print "\tmesh has %i cells and Lambda is %s" % (mesh0.num_cells(), Lambda0)
            if i == 0:
                mesh, Lambda = mesh0, set(Lambda0)
            else:
                # mesh, _ = create_joint_mesh([mesh0, mesh])
                if mesh.num_cells() < mesh0.num_cells():
                    mesh = mesh0
                Lambda = Lambda.union(set(Lambda0))
    print "=== FINAL mesh has %i cells and len(Lambda) is %i ===" % (mesh.num_cells(), len(Lambda))
    return mesh, Lambda

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_reference_setup('.')
