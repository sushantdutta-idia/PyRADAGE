"""Core inclusion changes likelihood support, not just display coverage."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('astropy')
from astropy.io import fits
from astropy.wcs import WCS


@pytest.fixture(scope='module')
def core_case(tmp_path_factory):
    directory = tmp_path_factory.mktemp('included-core')
    script = Path(__file__).resolve().parents[1]/'examples/generate_radio_demo.py'
    spec = importlib.util.spec_from_file_location('core_policy_example', script)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generator.generate(directory, particles=64)
    config = json.loads((directory/'radio.json').read_text())
    config['inference'].update(replicates=1, mcmc_steps=1)
    with fits.open(directory/'mask.fits') as hdus:
        header = hdus[0].header.copy()
        core = np.zeros_like(hdus[0].data, dtype=np.uint8)
        core[14:17, 13:16] = 1
    fits.PrimaryHDU(core, header=header).writeto(directory/'core.fits')
    coordinate = WCS(header).celestial.pixel_to_world(14., 15.).icrs
    config['core'] = {'mask':'core.fits', 'policy':'include',
        'position_icrs_deg':[coordinate.ra.deg, coordinate.dec.deg],
        'description':'Synthetic detected core region included in the declared model.'}
    return directory, config, core.astype(bool)


@pytest.fixture(scope='module')
def included_core(core_case):
    from fr2dynage.radio_inference import compute_radio_inference
    directory, config, _ = core_case
    path = directory/'included.json'
    path.write_text(json.dumps(config))
    return compute_radio_inference(path)


def test_included_core_has_transport_values_and_enters_model_regions(core_case, included_core):
    _, _, core = core_case
    result = included_core
    assert np.all(result.maps['AGE_VALID'][core])
    assert np.all(np.isfinite(result.maps['TRANSPORT'][core]))
    assert np.all(result.maps['TRANSPORT'][core] > 0)
    assert not np.any(result.maps['AGE_FLAGS'][core] & 4)
    assert np.any(result.maps['REGION_ID'][core] > 0)
    assert result.metadata['core']['policy'] == 'include'
    assert result.metadata['core']['modelled_pixels'] == 9
    assert result.metadata['source_coverage']['core_age_valid_pixels'] == 9
    assert 'including the core' in result.metadata['map_statistics']['scope']
    assert not any('do not enter' in value for value in result.metadata['core']['warnings'])


def test_included_core_preview_has_no_circular_mask_or_grey_overlay(included_core, tmp_path, monkeypatch):
    from matplotlib.contour import QuadContourSet
    import fr2dynage.plotting as plotting
    from fr2dynage.radio_inference import render_radio_preview
    observed=[]
    def capture(figure, path, **kwargs):
        assert not any(isinstance(artist, QuadContourSet) for artist in figure.findobj())
        sky_axes=[axis for axis in figure.axes if hasattr(axis, 'coords')]
        assert len(sky_axes) == 4
        assert [len(axis.images) for axis in sky_axes] == [1, 1, 1, 1]
        assert 'Grey:' not in figure._suptitle.get_text()
        observed.append(True)
        Path(path).write_bytes(b'captured preview')
        return [Path(path)]
    monkeypatch.setattr(plotting, 'save_publication_figure', capture)
    render_radio_preview(included_core, tmp_path/'included.png')
    assert len(observed) == 3


def test_included_core_fits_declares_modelled_clock(included_core, tmp_path):
    from fr2dynage.radio_inference import write_radio_products
    write_radio_products(included_core, tmp_path/'included.fits', tmp_path/'summary.json')
    with fits.open(tmp_path/'included.fits', checksum=True) as hdus:
        assert hdus[0].header['COREAGE'] == 'MODELLED'
        assert hdus[0].header['COREPOL'] == 'INCLUDE'
        core=hdus['CORE'].data.astype(bool)
        assert np.all(np.isfinite(hdus[0].data[core]))
        for hdu in hdus:
            assert hdu.verify_checksum() == 1


@pytest.mark.parametrize('policy', ['drop', None, True, ['include']])
def test_invalid_core_policy_is_rejected_before_sampling(core_case, policy, monkeypatch):
    import fr2dynage.radio_inference as module
    directory, base, _ = core_case
    config=deepcopy(base);config['core']['policy']=policy
    path=directory/'invalid-policy.json';path.write_text(json.dumps(config))
    def sampler(*args, **kwargs):
        pytest.fail('invalid core policy must be rejected before sampling')
    monkeypatch.setattr(module, 'tempered_smc', sampler)
    with pytest.raises(ValueError, match='core.policy'):
        module.compute_radio_inference(path)


def test_legacy_missing_policy_preserves_explicit_separate_core(core_case):
    from fr2dynage.radio_inference import compute_radio_inference
    directory, base, core=core_case
    config=deepcopy(base);del config['core']['policy']
    path=directory/'legacy-core.json';path.write_text(json.dumps(config))
    result=compute_radio_inference(path)
    assert result.metadata['core']['policy'] == 'separate'
    assert not np.any(result.maps['AGE_VALID'][core])
    assert np.all(np.isnan(result.maps['TRANSPORT'][core]))
    assert np.all(result.maps['AGE_FLAGS'][core] & 4)
