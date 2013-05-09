from __future__ import division
import logging

from spuq.math_utils.multiindex import Multiindex
from spuq.utils.timing import timing

try:
    from dolfin import (interactive, project, errornorm, norm)
    from spuq.application.egsz.sampling import (get_projection_basis, compute_direct_sample_solution,
                                                compute_parametric_sample_solution)
    from spuq.fem.fenics.fenics_utils import error_norm
except Exception, e:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# module logger
logger = logging.getLogger(__name__)


class MCCache(object):
    pass

def run_mc(err, w, pde, A, coeff_field, mesh0, ref_maxm, MC_N, MC_HMAX, param_sol_cache=None, direct_sol_cache=None, stored_rv_samples=None):
    # create reference mesh and function space
    sub_spaces = w[Multiindex()].basis.num_sub_spaces
    degree = w[Multiindex()].basis.degree
    projection_basis = get_projection_basis(mesh0, mesh_refinements=0, degree=degree, sub_spaces=sub_spaces)
    logger.info("projection_basis dim = %i \t hmin of mi[0] = %s, reference mesh = (%s, %s)", projection_basis.dim, w[Multiindex()].basis.minh, projection_basis.minh, projection_basis.maxh)

    # get realization of coefficient field
    err_L2, err_H1 = 0, 0

    # setup caches for sample solutions
#    param_sol_cache = None #param_sol_cache or MCCache()
#    direct_sol_cache = None #direct_sol_cache or MCCache()
    logger.info("---- MC caches %s/%s ----", param_sol_cache, direct_sol_cache)
    # main MC loop
    for i in range(MC_N):
        logger.info("---- MC Iteration %i/%i ----", i + 1 , MC_N)
        # create new samples if required or reuse existing samples
        if stored_rv_samples is None or len(stored_rv_samples) <= i:
            sample_rvs = coeff_field.sample_rvs()
            RV_samples = [sample_rvs[j] for j in range(max(w.max_order, ref_maxm))]
            if stored_rv_samples is not None:
                stored_rv_samples.append(RV_samples)
        else:
            RV_samples = stored_rv_samples[i]        
        logger.info("-- RV_samples: %s", RV_samples)
        # evaluate solutions
        with timing(msg="parameteric sample solution", logfunc=logger.info):
            sample_sol_param = compute_parametric_sample_solution(RV_samples, coeff_field, w, projection_basis, param_sol_cache)
        with timing(msg="direct sample solution", logfunc=logger.info):
            sample_sol_direct = compute_direct_sample_solution(pde, RV_samples, coeff_field, A, ref_maxm, projection_basis, direct_sol_cache)
        # evaluate errors
        with timing(msg="L2_err_1", logfunc=logger.info):
            cerr_L2 = error_norm(sample_sol_param._fefunc, sample_sol_direct._fefunc, "L2")
        with timing(msg="H1A_err_1", logfunc=logger.info):
            cerr_H1 = error_norm(sample_sol_param._fefunc, sample_sol_direct._fefunc, pde.energy_norm)
#        cerr_H1 = errornorm(sample_sol_param._fefunc, sample_sol_direct._fefunc, "H1")
        logger.debug("-- current error L2 = %s    H1A = %s", cerr_L2, cerr_H1)
        err_L2 += 1.0 / MC_N * cerr_L2
        err_H1 += 1.0 / MC_N * cerr_H1
        
        if i + 1 == MC_N:
            # deterministic part
            with timing(msg="direct a0", logfunc=logger.info):
                sample_sol_direct_a0 = compute_direct_sample_solution(pde, RV_samples, coeff_field, A, 0, projection_basis, direct_sol_cache)
            with timing(msg="L2_err_2", logfunc=logger.info):
                L2_a0 = error_norm(sample_sol_param._fefunc, sample_sol_direct_a0._fefunc, "L2")
            with timing(msg="H1A_err_2", logfunc=logger.info):
                H1_a0 = error_norm(sample_sol_param._fefunc, sample_sol_direct_a0._fefunc, pde.energy_norm)
#            H1_a0 = errornorm(sample_sol_param._fefunc, sample_sol_direct_a0._fefunc, "H1")
            logger.debug("-- DETERMINISTIC error L2 = %s    H1A = %s", L2_a0, H1_a0)

            # stochastic part
            sample_sol_direct_am = sample_sol_direct - sample_sol_direct_a0
            logger.debug("-- STOCHASTIC norm L2 = %s    H1 = %s", sample_sol_direct_am.norm("L2"), sample_sol_direct_am.norm("H1"))

    logger.info("MC Error: L2: %s, H1A: %s", err_L2, err_H1)
    err.append((err_L2, err_H1, L2_a0, H1_a0))


def sample_error_mc(w, pde, A, coeff_field, mesh0, ref_maxm, MC_RUNS, MC_N, MC_HMAX, stored_rv_samples=None):
    # iterate MC
    err = []
    param_sol_cache = MCCache()
    direct_sol_cache = MCCache()
    for i in range(MC_RUNS):
        logger.info("---> MC RUN %i/%i  (with N=%i) <---", i + 1, MC_RUNS, MC_N)
        run_mc(err, w, pde, A, coeff_field, mesh0, ref_maxm, MC_N, MC_HMAX,
               param_sol_cache=param_sol_cache, direct_sol_cache=direct_sol_cache, stored_rv_samples=stored_rv_samples)
    #print "evaluated errors (L2,H1):", err
    L2err = sum([e[0] for e in err]) / len(err)
    H1err = sum([e[1] for e in err]) / len(err)
    L2err_a0 = sum([e[2] for e in err]) / len(err)
    H1err_a0 = sum([e[3] for e in err]) / len(err)
    logger.info("average MC ERRORS: L2 = %s   H1 = %s    [deterministic part L2 = %s    H1 = %s]", L2err, H1err, L2err_a0, H1err_a0)
    return L2err, H1err, L2err_a0, H1err_a0, len(err)
