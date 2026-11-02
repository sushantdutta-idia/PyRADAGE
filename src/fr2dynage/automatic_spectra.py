"""Shared injection-index fits with beam and shared-calibration covariance.

Independent per-region ages and amplitudes supply spectral shape information;
no core coordinate, traced flow, morphology label or dynamical history is
invented. Local information errors and source ages remain conditional.
"""
import math

import numpy as np
from scipy.linalg import cholesky, null_space, solve_triangular
from scipy.optimize import least_squares

from .advanced_spectra import spectral_log_emissivity
from .maps_io import MapInputError, science_image
from .resolved_spectra import fit_resolved_spectra
from .spectral_maps import _sigma_ln


def fit_shared_injection(prepared, redshift, *, model='jp_tribble', field_ug=3.,
                         max_regions=32, cancel_event=None):
    """Fit one shared alpha plus independent regional radiative clocks.

    The reference field fixes the age coordinate for this shape fit. Field
    changes are exactly age-coordinate scalings in these kernels; the final
    pixel fits are recomputed at the adopted field and declared age bounds.
    All normalizations are eliminated by GLS, including global calibration
    covariance. A finite best fit is not a measured injection index or age.
    """
    def cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError('Analysis cancelled')
    cancelled()
    frequencies = np.asarray(prepared.frequencies_mhz, dtype=float)
    flux, rms = prepared.flux_cube, prepared.rms_cube
    source = prepared.source_mask.astype(bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = flux/rms
    detected = np.isfinite(snr) & (snr >= 5.) & (flux > 0) & np.isfinite(rms) & (rms > 0)
    valid = source & (detected.sum(axis=0) >= 4)
    yy, xx = np.nonzero(valid)
    if not len(yy):
        raise MapInputError('No source pixel has four usable detections for spectral fitting')
    from astropy.wcs.utils import proj_plane_pixel_area
    _, beam = science_image(prepared.reference)
    area = abs(float(proj_plane_pixel_area(prepared.reference.wcs)))
    step = max(1, int(math.ceil(beam['bmaj_deg']/math.sqrt(area))))
    score = np.sort(np.where(detected, snr, -np.inf), axis=0)[-4]
    groups = {}
    for y,x in zip(yy,xx):
        key = (int(y//step), int(x//step))
        if key not in groups or score[y,x] > score[groups[key]]:
            groups[key]=(int(y),int(x))
    positions = np.array(list(groups.values()),dtype=int)
    # Deterministic spatial coverage, not a random strongest-pixel cut.
    if len(positions)>max_regions:
        positions=positions[np.linspace(0,len(positions)-1,max_regions,dtype=int)]
    ry,rx=positions.T
    use=detected[:,ry,rx].T
    rid,fid=np.nonzero(use)
    observed=flux[fid,ry[rid],rx[rid]]
    thermal=_sigma_ln(rms[fid,ry[rid],rx[rid]]/observed)
    sky=prepared.reference.wcs.pixel_to_world(rx,ry)
    origin=prepared.reference.wcs.pixel_to_world(float(rx[0]),float(ry[0]))
    east,north=origin.spherical_offsets_to(sky)
    offset=np.column_stack((east.deg,north.deg))
    delta=offset[:,None,:]-offset[None,:,:]
    spatial=np.exp(-np.einsum('...i,ij,...j->...',delta,np.linalg.inv(beam['covariance_deg2']),delta)/4.)
    covariance=spatial[rid[:,None],rid[None,:]]*(fid[:,None]==fid[None,:])*thermal[:,None]*thermal[None,:]
    cal=_sigma_ln(np.asarray(prepared.calibration_fractions))
    corr=np.asarray(prepared.calibration_correlation)
    covariance+=corr[fid[:,None],fid[None,:]]*cal[fid[:,None]]*cal[fid[None,:]]
    covariance+=np.eye(len(rid))*1e-12
    condition=float(np.linalg.cond(covariance))
    if not math.isfinite(condition) or condition>1e12:
        raise MapInputError('Automatic regional covariance is ill-conditioned; check noise/calibration metadata')
    root=cholesky(covariance,lower=True)
    invroot=solve_triangular(root,np.eye(len(rid)),lower=True)
    amplitudes=np.eye(len(positions))[rid]
    design=invroot@amplitudes
    projector=null_space(design.T).T@invroot
    data=projector@np.log(observed)
    dof=len(rid)-2*len(positions)-1
    if dof<1:
        raise MapInputError('Too few independent spectral measurements for shared-index fitting')

    def residual(parameters):
        cancelled()
        alpha=float(parameters[0]);ages=np.expm1(parameters[1:])
        shape=spectral_log_emissivity(frequencies,ages,field_ug,alpha,redshift,model=model)
        return data-projector@shape[rid,fid]

    solutions=[]
    for start_alpha in [.55,.75,.95]:
        cancelled()
        independent=fit_resolved_spectra(frequencies,flux[:,ry,rx][:,None,:],rms[:,ry,rx][:,None,:],
            prepared.calibration_fractions,prepared.calibration_correlation,
            np.ones((1,len(positions))),field_ug,redshift,start_alpha,model=model,
            age_bounds_myr=(0.,3000.),age_grid_size=121,cancel_event=cancel_event)
        ages=independent['maps']['SPECTRAL_AGE'][0]
        ages=np.where(np.isfinite(ages),ages,10.)
        fit=least_squares(residual,np.r_[start_alpha,np.log1p(ages)],
            bounds=(np.r_[.5,np.zeros(len(ages))],np.r_[1.3,np.full(len(ages),np.log1p(3000.))]),
            max_nfev=500,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        if np.all(np.isfinite(fit.fun)):
            solutions.append(fit)
    if not solutions:
        raise MapInputError('No finite shared-index spectral fit')
    best=min(solutions,key=lambda result:float(result.fun@result.fun))
    alpha=float(best.x[0]);chi2=float(best.fun@best.fun)
    sigma=None
    singular=np.linalg.svd(best.jac,compute_uv=False)
    information_condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
    if information_condition is not None and information_condition<1e6:
        try:
            covariance_local=np.linalg.inv(best.jac.T@best.jac)
            if covariance_local[0,0]>0:
                sigma=float(np.sqrt(covariance_local[0,0]))
        except np.linalg.LinAlgError:
            pass
    flags=[]
    if not best.success:flags.append('optimizer_not_converged')
    if alpha<.505 or alpha>1.295:flags.append('injection_index_search_boundary')
    ages=np.expm1(best.x[1:])
    if np.any(ages<1e-5) or np.any(ages>2997.):
        flags.append('regional_age_search_boundary_local_errors_nonregular')
    if chi2/dof>4.:flags.append('poor_shared_spectral_fit')
    if sigma is None or sigma>.15:flags.append('injection_index_weak_or_singular_local_information')
    width=max(.05,sigma) if sigma is not None and math.isfinite(sigma) else .3
    return {'model':model,'alpha_inj':alpha,'alpha_local_sigma':sigma,
        'alpha_sensitivity':[max(.5,alpha-width),min(1.3,alpha+width)],
        'chi2':chi2,'degrees_of_freedom':dof,'reduced_chi2':chi2/dof,
        'status':'diagnostic_only' if flags else 'conditional_shared_index_fit','flags':flags,
        'optimizer_success':bool(best.success),'optimizer_message':str(best.message),
        'regional_pixels_xy':positions[:,::-1].tolist(),'regional_best_ages_myr':np.expm1(best.x[1:]).tolist(),
        'reference_field_ug':float(field_ug),'covariance_condition_number':condition,
        'local_jacobian_condition_number':information_condition,
        'scope':'Shared injection index and independent regional clocks/GLS amplitudes; full Gaussian beam-thermal and shared band-calibration covariance. Local index error is an asymptotic diagnostic, not a marginalized interval or independent field measurement. Source mixing, uv coverage and physical histories are not inferred.'}
