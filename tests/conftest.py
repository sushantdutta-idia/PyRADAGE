"""Keep bundle tests inexpensive; test_plotting checks actual 600 dpi output."""
import pytest


@pytest.fixture(autouse=True, scope="module")
def bounded_bundle_raster_resolution(request):
    if request.node.path.name == "test_plotting.py":
        yield
        return
    import fr2dynage.plotting as plotting
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plotting, "RASTER_DPI", 72)
        yield
