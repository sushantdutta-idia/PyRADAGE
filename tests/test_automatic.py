"""Image/redshift workflow, real spectral fits, and atomic failure checks."""
import importlib.util
import json
from pathlib import Path
import threading

import numpy as np
import pytest
from astropy.io import fits

from fr2dynage.automatic import run_automatic
from fr2dynage.reliability import verify_analysis_manifest


@pytest.fixture(scope='module')
def automatic_demo(tmp_path_factory):
    script=Path(__file__).resolve().parents[1]/'examples/generate_automatic_demo.py'
    spec=importlib.util.spec_from_file_location('automatic_demo_generator',script)
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    parent=tmp_path_factory.mktemp('automatic-example');folder=parent/'images'
    generator.generate(folder)
    return sorted(folder.glob('*.fits')),parent


@pytest.fixture(scope='module')
def automatic_result(automatic_demo):
    paths,parent=automatic_demo
    return run_automatic(paths,.15,parent/'result',field_t=3e-10,max_regions=8)


def test_images_and_redshift_produce_one_summary_pair_and_auditable_maps(automatic_result):
    folder=Path(automatic_result['output_dir'])
    assert Path(automatic_result['summary_pdf']).read_bytes().startswith(b'%PDF')
    assert Path(automatic_result['summary_png']).read_bytes().startswith(b'\x89PNG')
    assert len(list(folder.glob('*.png')))==1
    assert len(list(folder.glob('*.pdf')))==1
    assert verify_analysis_manifest(folder,check_inputs=True)['passed']
    metadata=json.loads((folder/'analysis.json').read_text())
    assert metadata['global_dynamical_age']['mean_myr'] is None
    assert metadata['field']['adopted_field_t']==3e-10
    assert metadata['preparation']['fixed_circular_exclusion'] is False
    for model in ['jp_tribble','ci']:
        with fits.open(folder/f'resolved_{model}.fits',checksum=True) as hdus:
            assert all(h.verify_checksum()==1 and h.verify_datasum()==1 for h in hdus)
            assert hdus[0].header['BINPUT']=='USERFIX'
            assert hdus[0].header['COREPOL']=='INCLUDE'
            assert hdus[0].header['GLOBAGE']=='UNAVAILABLE'
            assert np.isfinite(hdus[0].data[32,32])
            assert 'AGE_B16' not in hdus
            saved=json.loads((folder/f'resolved_{model}.json').read_text())
            assert json.loads(bytes(hdus['PROVENANCE'].data['JSON_UTF8'][0]))==saved
        assert metadata['shared_spectral_fits'][model]['alpha_inj']>=.5
    assert not list(folder.parent.glob('.result.*'))
    assert '.result.' not in (folder/'prepared/preparation.json').read_text()


def test_existing_results_and_inputs_are_preserved(automatic_demo,automatic_result):
    paths,parent=automatic_demo
    with pytest.raises(FileExistsError):run_automatic(paths,.15,parent/'result',field_t=3e-10)
    with pytest.raises((ValueError,FileExistsError)):
        run_automatic(paths,.15,paths[0].parent.parent,field_t=3e-10)


def test_cancel_leaves_no_partial_analysis(automatic_demo):
    paths,parent=automatic_demo
    event=threading.Event();event.set()
    with pytest.raises(InterruptedError):
        run_automatic(paths,.15,parent/'cancelled',field_t=3e-10,cancel_event=event)
    assert not (parent/'cancelled').exists()


@pytest.mark.parametrize('redshift',[True,0.,-.1,float('nan')])
def test_bad_redshift_is_rejected_before_backend_or_data(redshift,tmp_path):
    with pytest.raises(ValueError,match='Redshift'):
        run_automatic([],redshift,tmp_path/'out')
