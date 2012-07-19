from __future__ import division
import logging

from spuq.math_utils.multiindex import Multiindex

try:
    from dolfin import (interactive, project, errornorm, norm)
    from spuq.application.egsz.sampling import (get_projection_basis, compute_direct_sample_solution,
                                                compute_parametric_sample_solution, setup_vector)
except Exception, e:
    print "FEniCS has to be available"

# module logger
logger = logging.getLogger(__name__)

def run_mc(err, w, pde, A, coeff_field, mesh0, ref_maxm, MC_N, MC_HMAX):
    # create reference mesh and function space
    projection_basis = get_projection_basis(mesh0, maxh=min(w[Multiindex()].basis.minh / 4, MC_HMAX))
    logger.debug("hmin of mi[0] = %s, reference mesh = (%s, %s)", w[Multiindex()].basis.minh, projection_basis.minh, projection_basis.maxh)

    # get realization of coefficient field
    err_L2, err_H1 = 0, 0
    for i in range(MC_N):
        logger.info("---- MC Iteration %i/%i ----", i + 1 , MC_N)
        RV_samples = coeff_field.sample_rvs()
        logger.info("-- RV_samples: %s", [RV_samples[j] for j in range(w.max_order)])
        sample_sol_param = compute_parametric_sample_solution(RV_samples, coeff_field, w, projection_basis)
        sample_sol_direct = compute_direct_sample_solution(pde, RV_samples, coeff_field, A, ref_maxm, projection_basis)
        cerr_L2 = errornorm(sample_sol_param._fefunc, sample_sol_direct._fefunc, "L2")
        cerr_H1 = errornorm(sample_sol_param._fefunc, sample_sol_direct._fefunc, "H1")
        logger.debug("-- current error L2 = %s    H1 = %s", cerr_L2, cerr_H1)
        err_L2 += 1.0 / MC_N * cerr_L2
        err_H1 += 1.0 / MC_N * cerr_H1
        
        if i + 1 == MC_N:
            # deterministic part
            sample_sol_direct_a0 = compute_direct_sample_solution(pde, RV_samples, coeff_field, A, 0, projection_basis)
            L2_a0 = errornorm(sample_sol_param._fefunc, sample_sol_direct_a0._fefunc, "L2")
            H1_a0 = errornorm(sample_sol_param._fefunc, sample_sol_direct_a0._fefunc, "H1")
            logger.debug("-- DETERMINISTIC error L2 = %s    H1 = %s", L2_a0, H1_a0)

            # stochastic part
            sample_sol_direct_am = sample_sol_direct - sample_sol_direct_a0
            logger.debug("-- STOCHASTIC norm L2 = %s    H1 = %s", sample_sol_direct_am.norm("L2"), sample_sol_direct_am.norm("H1"))

    # remove cached expressions from coeff_field
    try:
        del coeff_field.A
        # TODO: for some strange reason its not possible to delete coeff_field.Am
        coeff_field.Am = None
    except:
        pass
    
    logger.info("MC Error: L2: %s, H1: %s", err_L2, err_H1)
    err.append((err_L2, err_H1, L2_a0, H1_a0))


def sample_error_mc(w, pde, A, coeff_field, mesh0, ref_maxm, MC_RUNS, MC_N, MC_HMAX):
    # iterate MC
    err = []
    for i in range(MC_RUNS):
        logger.info("---> MC RUN %i/%i <---", i + 1, MC_RUNS)
        run_mc(err, w, pde, A, coeff_field, mesh0, ref_maxm, MC_N, MC_HMAX)
    
    #print "evaluated errors (L2,H1):", err
    L2err = sum([e[0] for e in err]) / len(err)
    H1err = sum([e[1] for e in err]) / len(err)
    L2err_a0 = sum([e[2] for e in err]) / len(err)
    H1err_a0 = sum([e[3] for e in err]) / len(err)
    logger.info("average MC ERRORS: L2 = %s   H1 = %s    [deterministic part L2 = %s    H1 = %s]", L2err, H1err, L2err_a0, H1err_a0)
    return L2err, H1err, L2err_a0, H1err_a0
