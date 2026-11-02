from __future__ import annotations

import pytest

from fr2dynage.radio_classes import RADIO_CLASSES, normalize_radio_class, require_fr2_class


@pytest.mark.parametrize("supplied, expected", [
    ("FR-I", "fr1"), ("FR II", "fr2"), ("FR–II", "fr2"), ("FR 0", "fr0"),
    ("WAT", "wat"), ("wide angle tail", "wat"), ("Narrow-angle tailed", "nat"),
    ("Head-tail", "head_tail"), ("HYMORS", "hybrid"), ("CSS", "css"),
    ("GPS", "gps"), ("Compact symmetric object", "cso"), (" Unclassified ", "unknown"),
])
def test_radio_class_aliases(supplied, expected) -> None:
    assert normalize_radio_class(supplied) == expected


@pytest.mark.parametrize("supplied", [None, True, 2, "", " ", "spelling-error"])
def test_unknown_class_labels_do_not_silently_select_a_model(supplied) -> None:
    with pytest.raises(ValueError, match="radio_class"):
        normalize_radio_class(supplied)


def test_all_classes_have_kinematic_workflows_but_only_fr2_has_kda() -> None:
    for label, details in RADIO_CLASSES.items():
        assert normalize_radio_class(label) == label
        assert details.map_methods == ("expansion", "flow")
        assert details.kda_supported is (label == "fr2")
        if label == "fr2":
            assert require_fr2_class(label) is None
        else:
            with pytest.raises(ValueError, match="map-age"):
                require_fr2_class(label)
