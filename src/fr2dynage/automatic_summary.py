"""One title-free publication figure for automatic spectral-clock products.

JP--Tribble radiative elapsed time and CI injection duration are deliberately
separate panels. A lower chi-square does not make their physical clocks
interchangeable, and no mixed-clock ``best dynamical age`` map is constructed.
"""
from __future__ import annotations

from pathlib import Path
import math
import os
import shutil
import tempfile

import numpy as np

from .plotting import (AXIS_LABEL_SIZE, FONT_SIZE, LEGEND_SIZE, TICK_SIZE,
                       publication_figure, save_publication_figure)


SUMMARY_PANEL_DESCRIPTIONS = {
    'a': 'Reference-image brightness; the source footprint is retained including central emission.',
    'b': 'Conditional best-fit JP--Tribble radiative elapsed time, not a global dynamical lifetime.',
    'c': 'Conditional best-fit CI injection duration, a different physical clock from panel b.',
    'd': 'Conditional JP--Tribble upper95 profile endpoint; open profiles remain undefined.',
    'e': 'JP--Tribble reduced chi-square with the stated measurement/covariance model.',
    'f': 'JP--Tribble profile quality: no numerical fit, conditional/unconstrained, profile-constrained, or poor fit.',
}


def _real_image(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or raw.shape != shape:
        raise ValueError(f'{name} must be a real image matching the reference shape')
    return np.asarray(raw,dtype=float)


def _binary(value, shape, name):
    raw=np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in 'biuf' or not np.all((raw==0)|(raw==1)):
        raise ValueError(f'{name} must be a binary image matching the reference shape')
    return raw.astype(bool)


def _model_maps(fitted, model, shape):
    if not isinstance(fitted,dict) or model not in fitted or not isinstance(fitted[model],dict):
        raise ValueError(f'fitted requires the {model} resolved result')
    maps=fitted[model].get('maps')
    if not isinstance(maps,dict):
        raise ValueError(f'{model} requires a maps dictionary')
    required=['SPECTRAL_AGE','AGE_UPPER95','REDCHI2','SPEC_VALID','AGE_CONSTRAINED','AGE_FLAGS']
    missing=set(required)-set(maps)
    if missing:
        raise ValueError(f'{model} is missing required maps: {sorted(missing)}')
    values={key:_real_image(maps[key],shape,f'{model}.{key}')
            for key in ['SPECTRAL_AGE','AGE_UPPER95','REDCHI2']}
    values['SPEC_VALID']=_binary(maps['SPEC_VALID'],shape,f'{model}.SPEC_VALID')
    values['AGE_CONSTRAINED']=_binary(maps['AGE_CONSTRAINED'],shape,f'{model}.AGE_CONSTRAINED')
    flags=np.asarray(maps['AGE_FLAGS'])
    if (flags.shape!=shape or flags.dtype.kind not in 'iu' or
        np.any(flags<0) or np.any(flags>65535)):
        raise ValueError(f'{model}.AGE_FLAGS must be an integer quality-bit image')
    values['AGE_FLAGS']=flags.astype(np.uint16)
    valid=values['SPEC_VALID']
    if np.any(~np.isfinite(values['SPECTRAL_AGE'][valid])) or np.any(values['SPECTRAL_AGE'][valid]<0):
        raise ValueError(f'{model} valid ages must be finite and nonnegative')
    if np.any(values['AGE_CONSTRAINED'] & ~valid):
        raise ValueError(f'{model} constrained ages must also be numerically fitted')
    return values


def _beam_and_scale(axis,reference,shape):
    """Mark the restoring beam and a purely angular scale when well defined.

    This does not assume a cosmology or invent a source/core aperture. The
    local celestial projection matrix is adequate for a compact source cutout.
    """
    from matplotlib.patches import Ellipse
    matrix=np.asarray(reference.wcs.pixel_scale_matrix,dtype=float)
    if matrix.shape!=(2,2) or not np.all(np.isfinite(matrix)):
        return
    ny,nx=shape
    horizontal_arcsec=float(np.linalg.norm(matrix[:,0])*3600.)
    if math.isfinite(horizontal_arcsec) and horizontal_arcsec>0 and nx>=12:
        desired=.22*nx*horizontal_arcsec
        order=10.**math.floor(math.log10(desired))
        length=max(value*order for value in [.1,.2,.5,1.,2.,5.,10.] if value*order<=desired)
        pixels=length/horizontal_arcsec
        x0,y0=.08*(nx-1),.9*(ny-1)
        axis.plot([x0,x0+pixels],[y0,y0],color='black',linewidth=3.,zorder=8)
        axis.text(x0+.5*pixels,y0+.025*ny,f'{length:g}″',ha='center',va='bottom',
                  fontsize=LEGEND_SIZE,color='black',zorder=8,
                  bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1.})
    header=getattr(reference,'header',{})
    try:
        major,minor=float(header['BMAJ']),float(header['BMIN'])
        angle=math.radians(float(header.get('BPA',0.)))
        if not np.isfinite([major,minor,angle]).all() or min(major,minor)<=0:
            return
        direction=np.array([math.sin(angle),math.cos(angle)])
        other=np.array([math.cos(angle),-math.sin(angle)])
        beam=(np.outer(direction,direction)*major**2+np.outer(other,other)*minor**2)
        inverse=np.linalg.inv(matrix)
        projected=inverse@beam@inverse.T
        eigenvalues,eigenvectors=np.linalg.eigh(projected)
        if np.any(eigenvalues<=0):
            return
        width,height=np.sqrt(eigenvalues[::-1])
        if not np.isfinite([width,height]).all() or min(width,height)<=0 or max(width,height)>.25*min(shape):
            return
        vector=eigenvectors[:,-1]
        rotation=math.degrees(math.atan2(vector[1],vector[0]))
        axis.add_patch(Ellipse((.88*(nx-1),.1*(ny-1)),width,height,angle=rotation,
                              facecolor='white',edgecolor='black',linewidth=1.6,zorder=8))
    except (KeyError,TypeError,ValueError,np.linalg.LinAlgError,OverflowError):
        return


def _build_summary(reference,intensity,models,mask):
    from matplotlib.colors import AsinhNorm, BoundaryNorm, ListedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # Explicit panel spacing plus image-anchored colour bars avoids the long
    # bars/large gaps produced when constrained layout shrinks equal-aspect
    # WCS images independently of their original subplot rectangles.
    # A portrait two-column figure retains roughly 10-point labels when
    # reduced to a typical seven-inch journal two-column width.
    columns=2
    figure=publication_figure(figsize=(12,15),layout='none')
    grid=figure.add_gridspec(3,columns,left=.11,right=.91,bottom=.07,top=.95,wspace=.6,hspace=.3)
    tjp,ci=models['jp_tribble'],models['ci']
    valid=mask&tjp['SPEC_VALID']
    quality=np.full(mask.shape,np.nan)
    quality[mask & ~tjp['SPEC_VALID']]=1.
    quality[valid]=2.
    quality[mask&tjp['AGE_CONSTRAINED']]=3.
    quality[valid&((tjp['AGE_FLAGS']&64)!=0)]=4.
    quality_cmap=ListedColormap(['#bdbdbd','#e6ab02','#1b9e77','#d95f02'])
    panels=[
        (np.where(mask,intensity*1000.,np.nan),'Brightness (mJy beam⁻¹)','magma','a'),
        (np.where(valid,tjp['SPECTRAL_AGE'],np.nan),'JP–Tribble time (Myr)','viridis','b'),
        (np.where(mask&ci['SPEC_VALID'],ci['SPECTRAL_AGE'],np.nan),'CI injection duration (Myr)','viridis','c'),
        (np.where(valid,tjp['AGE_UPPER95'],np.nan),'JP–Tribble upper95 (Myr)','magma','d'),
        (np.where(valid,tjp['REDCHI2'],np.nan),'JP–Tribble reduced χ²','cividis','e'),
        (quality,'JP–Tribble profile quality',quality_cmap,'f')]
    for index,(values,label,cmap,letter) in enumerate(panels,1):
        axis=figure.add_subplot(grid[(index-1)//columns,(index-1)%columns],projection=reference.wcs)
        norm=None
        if index==1:
            finite=values[np.isfinite(values)]
            if finite.size and float(finite.max())>0:
                peak=float(finite.max())
                norm=AsinhNorm(linear_width=.01*peak,vmin=min(float(finite.min()),0.),vmax=peak)
        if index==6:
            norm=BoundaryNorm(np.arange(.5,5.,1.),4)
        shown=axis.imshow(values,origin='lower',cmap=cmap,norm=norm,interpolation='nearest')
        axis.text(.035,.035,f'({letter})',transform=axis.transAxes,ha='left',va='bottom',
                  fontsize=FONT_SIZE,color='black',fontweight='bold',zorder=10,
                  bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2.})
        for coordinate,name in zip(axis.coords,['Right ascension','Declination']):
            coordinate.set_axislabel(name,fontsize=AXIS_LABEL_SIZE)
            coordinate.set_ticklabel(size=TICK_SIZE,exclude_overlapping=True)
            coordinate.set_ticks(number=3)
        if (index-1)%columns:
            # Every panel uses the same sky extent. Shared declination labels
            # leave room for the preceding panel's physical colour-bar label.
            axis.coords[1].set_axislabel('')
            axis.coords[1].set_ticklabel_visible(False)
        if np.any(np.isfinite(values)):
            bar_axis=inset_axes(axis,width='5%',height='100%',loc='lower left',
                               bbox_to_anchor=(1.04,0,1,1),bbox_transform=axis.transAxes,borderpad=0)
            colorbar=figure.colorbar(shown,cax=bar_axis)
            colorbar.set_label(label,fontsize=AXIS_LABEL_SIZE)
            if index==1 and norm is not None:
                peak=float(np.nanmax(values))
                ticks=[0.,.1*peak,peak]
                colorbar.set_ticks(ticks,labels=[f'{tick:.2g}' for tick in ticks])
            if index==6:
                colorbar.set_ticks([1,2,3,4],labels=['No fit','Conditional','Constrained','Poor fit'])
        else:
            message='No finite upper limits' if index==4 else 'No fitted values'
            axis.text(.5,.5,message,ha='center',va='center',transform=axis.transAxes,
                      fontsize=FONT_SIZE,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
            # Keep panel semantics visible even when no colour bar is possible.
            axis.text(.5,.12,label,ha='center',va='center',transform=axis.transAxes,
                      fontsize=LEGEND_SIZE)
        if index==1:
            _beam_and_scale(axis,reference,mask.shape)
    return figure


def render_automatic_summary(reference,intensity_jy_per_beam,fitted,output_path,*,source_mask=None,overwrite=False):
    """Publish one combined PNG/PDF/SVG figure without top titles.

    The source footprint is not cut at the core. Numerical best fits remain
    visible with their quality class; no unsupported fit is promoted to a
    measured age. All input arrays are read-only from this function's view.
    Publication is staged and collision-checked across all formats.
    """
    shape=np.asarray(reference.data).shape
    if len(shape)!=2 or min(shape)<1 or not hasattr(reference,'wcs'):
        raise ValueError('reference requires a nonempty two-dimensional celestial image')
    intensity=_real_image(intensity_jy_per_beam,shape,'intensity_jy_per_beam')
    mask=(_binary(source_mask,shape,'source_mask') if source_mask is not None else np.isfinite(intensity))
    models={name:_model_maps(fitted,name,shape) for name in ['jp_tribble','ci']}
    output=Path(output_path).expanduser().resolve()
    targets={suffix:output.with_suffix('.'+suffix) for suffix in ['png','pdf','svg']}
    protected=Path(reference.path).resolve() if getattr(reference,'path',None) else None
    for path in targets.values():
        if protected is not None and path==protected:
            raise ValueError('summary must not overwrite its input image')
        if path.exists() and (not overwrite or not path.is_file()):
            raise FileExistsError(f'{path} exists; choose a new summary name')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.age-summary-',dir=output.parent))
    published=[]
    backups={}
    figure=None
    try:
        figure=_build_summary(reference,intensity,models,mask)
        generated=save_publication_figure(figure,stage/'summary.png',formats=('png','pdf','svg'),clear=True)
        for fmt,target in targets.items():
            if overwrite and target.exists():
                backup=stage/('previous.'+fmt)
                shutil.copy2(target,backup)
                backups[target]=backup
        for source,target in zip(generated,targets.values()):
            if overwrite:
                os.replace(source,target)
            else:
                os.link(source,target)
            published.append(target)
    except Exception:
        for target in reversed(published):
            if target in backups:
                os.replace(backups[target],target)
            else:
                target.unlink(missing_ok=True)
        raise
    finally:
        if figure is not None:
            figure.clear()
        shutil.rmtree(stage)
    return targets
