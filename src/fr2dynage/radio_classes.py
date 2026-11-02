"""Radio-class labels and the physical scope of available estimators.

Class labels are metadata, not a substitute for a model's physical assumptions.
The kinematic image workflows accept every class here when independent motion
and geometry constraints are supplied. The KA/KDA inverse solver remains an
FR II lobe model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RadioClass:
    label: str
    kda_supported: bool
    map_methods: tuple[str, ...] = ("expansion", "flow")
    guidance: str = "Use independent motion and geometry constraints for conditional kinematic maps."


RADIO_CLASSES: dict[str, RadioClass] = {
    "fr1": RadioClass("FR I", False),
    "fr2": RadioClass("FR II", True, guidance="KA/KDA is available for lobe-dominated FR II sources within its stated assumptions."),
    "fr0": RadioClass("FR 0", False, guidance="Resolve the emitting structure before interpreting a spatial kinematic clock."),
    "hybrid": RadioClass("Hybrid morphology", False, guidance="Analyse physically distinct components separately; only an FR II component can use KA/KDA."),
    "wat": RadioClass("Wide-angle tail", False, guidance="Use a measured path along the curved tail and an effective flow speed for a transport clock."),
    "nat": RadioClass("Narrow-angle tail", False, guidance="Use a measured path along the curved tail and an effective flow speed for a transport clock."),
    "head_tail": RadioClass("Head-tail", False, guidance="Use a measured path along the curved tail and an effective flow speed for a transport clock."),
    "compact": RadioClass("Compact radio galaxy", False, guidance="Supply one-sided core-to-component radius and physical component speed; convert measured double separation/rate to a consistent one-sided pair first."),
    "css": RadioClass("Compact steep-spectrum", False, guidance="Pair resolved component distances with independently measured motion; spectral class alone supplies no expansion law."),
    "gps": RadioClass("Gigahertz-peaked spectrum", False, guidance="Pair resolved component distances with independently measured motion; a spectral turnover is not an age measurement."),
    "cso": RadioClass("Compact symmetric object", False, guidance="Supply one-sided core-to-component radius and physical component speed; convert measured double separation/rate to a consistent one-sided pair first."),
    "diffuse": RadioClass("Diffuse radio galaxy", False),
    "other": RadioClass("Other radio galaxy", False),
    "unknown": RadioClass("Unclassified radio galaxy", False),
}


def _token(value: str) -> str:
    return "".join(character for character in value.lower() if character not in " _-\u2010\u2011\u2012\u2013\u2014")


_ALIASES = {_token(key): key for key in RADIO_CLASSES}
_ALIASES.update({
    "fri": "fr1", "fanaroffrileyi": "fr1",
    "frii": "fr2", "fanaroffrileyii": "fr2",
    "fanaroffriley0": "fr0", "hymors": "hybrid", "hybridmorphology": "hybrid",
    "wideangletail": "wat", "wideangletailed": "wat",
    "narrowangletail": "nat", "narrowangletailed": "nat",
    "headtailed": "head_tail", "ht": "head_tail",
    "compactsteepspectrum": "css", "gigahertzpeakedspectrum": "gps",
    "compactsymmetricobject": "cso", "unclassified": "unknown",
})


def normalize_radio_class(value: str) -> str:
    """Return a canonical class label; unknown labels require explicit ``other``."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("radio_class must be a non-empty string")
    normalized = _ALIASES.get(_token(value.strip()))
    if normalized is None:
        choices = ", ".join(RADIO_CLASSES)
        raise ValueError(f"unknown radio_class {value!r}; choose one of {choices} (use 'other' when appropriate)")
    return normalized


def require_fr2_class(value: str) -> None:
    """Guard use of FR II dynamics rather than infer a universal model."""

    normalized = normalize_radio_class(value)
    if not RADIO_CLASSES[normalized].kda_supported:
        raise ValueError(
            f"KA/KDA dynamics require an FR II lobe; {RADIO_CLASSES[normalized].label} "
            "can use map-age with independently constrained motion and geometry"
        )
