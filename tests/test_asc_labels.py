"""ASC labels (spaclone 1.2.0): what spaclone writes names antibody-secreting cells (ASC) and nothing else changed.

The names of earlier releases still work wherever they are input (region kinds, mappings by cell type or
compartment), with identical results; no output on the synthetic example, and no built-in constant, says "plasma".
"""

import numpy as np
import pandas as pd
import pytest

import spaclone as spc
from spaclone import bcr, gex, sim
from spaclone.gex.profile import PROFILE_MARKERS, PROFILE_READOUTS

#: Former names that no output may contain (case-insensitive substrings).
FORMER = ("plasma",)


def _strings(obj, out, depth=0):
    """Every string a result holds: values, categories, column and index names, dict keys (nested)."""
    if depth > 6:
        return out
    if isinstance(obj, str):
        out.add(obj)
    elif isinstance(obj, pd.DataFrame):
        out.update(map(str, obj.columns))
        _strings(obj.index, out, depth + 1)
        for c in obj.columns:
            _strings(obj[c], out, depth + 1)
    elif isinstance(obj, pd.Series | pd.Index):
        if isinstance(obj.dtype, pd.CategoricalDtype):
            out.update(map(str, obj.cat.categories) if isinstance(obj, pd.Series) else map(str, obj.categories))
        elif obj.dtype == object or pd.api.types.is_string_dtype(obj.dtype):
            out.update(str(v) for v in pd.unique(obj.dropna()) if isinstance(v, str))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            _strings(v, out, depth + 1)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _strings(v, out, depth + 1)
    return out


def _former(strings):
    return sorted(s for s in strings if any(f in s.lower() for f in FORMER))


def test_asc_kind_and_old_spellings_of_region_kinds():
    a = spc.datasets.synthetic_example().gex
    ref = a.copy()
    tab = gex.call_regions(ref, kind="asc")
    for spelling in ("ASC", "plasma_cell", "pc", "plasma"):  # the kind's names before 1.2 give the same call
        b = a.copy()
        pd.testing.assert_frame_equal(gex.call_regions(b, kind=spelling), tab)
        pd.testing.assert_frame_equal(b.obs, ref.obs)
        assert b.uns["spaclone"] == ref.uns["spaclone"]
    assert set(ref.obs["region"].dropna().astype(str)) == {"ASC", "other"}


def test_synthetic_mappings_accept_the_old_names():
    new = {"molecules_per_cell": {"ASC": 25.0}, "cdna_by_type": {"ASC": 1.5}, "constant_per_cell": {"ASC": 35.0},
           "cdna_by_compartment": {"ASC focus": 1.9}}  # fmt: skip
    old = {"molecules_per_cell": {"plasma": 25.0}, "cdna_by_type": {"plasma": 1.5},
           "constant_per_cell": {"plasma": 35.0}, "cdna_by_compartment": {"plasma-cell focus": 1.9}}  # fmt: skip
    small = {"n_sinks": 2, "rng": 1}
    a, b = sim.synthetic_chip(**new, **small), sim.synthetic_chip(**old, **small)
    pd.testing.assert_frame_equal(a.air, b.air)
    pd.testing.assert_frame_equal(a.gex.obs, b.gex.obs)
    assert (a.gex.X != b.gex.X).nnz == 0
    assert a.meta["params"] == b.meta["params"] and a.meta["params"]["cdna_by_type"]["ASC"] == 1.5
    default = sim.synthetic_chip(**small)
    assert not default.air[["spot_id", "clone_id"]].equals(a.air[["spot_id", "clone_id"]])  # the mappings are used
    la = sim.synthetic_chip(legacy=True, molecules_per_cell={"ASC": 25.0}, **small)
    lb = sim.synthetic_chip(legacy=True, molecules_per_cell={"plasma": 25.0}, **small)
    pd.testing.assert_frame_equal(la.air, lb.air)


def test_synthetic_section_and_constants_say_asc():
    syn = spc.datasets.synthetic_example(sequences=True)
    found = set()
    _strings(syn.air, found)
    _strings(syn.gex.obs, found)
    _strings(syn.meta, found)
    assert not _former(found)
    assert "PC0" not in found and {"ASC", "ASC focus", "ASC0"} <= found
    # categories in the order of earlier releases ("ASC" where "plasma" sorted): colours and legends follow them
    compartments = ["T zone", "follicle", "germinal centre", "off tissue", "ASC focus", "red pulp"]
    assert list(syn.gex.obs["cell_type"].cat.categories) == ["B", "GC B", "T", "none", "other", "ASC"]
    assert list(syn.gex.obs["compartment"].cat.categories) == compartments
    constants = [PROFILE_MARKERS, PROFILE_READOUTS, gex.REGION_LABELS, gex.REGION_MARKERS, gex.MARKERS,
                 sim.CDNA_BY_TYPE, sim.CDNA_BY_COMPARTMENT, sim.CONSTANT_PER_CELL]  # fmt: skip
    assert not _former(_strings(constants, set()))
    assert {"ASC", "ASC (Sdc1+)"} <= set(PROFILE_MARKERS["immune_mouse_no_ig"]) and "asc" in gex.REGION_LABELS


def test_workflow_and_report_on_the_synthetic_section_say_asc():
    syn = spc.datasets.synthetic_example(sequences=True)
    hot = bcr.hotspot_workflow(syn, n_perm=19, n_perm_region=19, rng=0)
    found = _strings({"module_maps": dict.fromkeys(hot.module_maps), "summary": hot.summary(), "repr": repr(hot),
                      "regions": hot.regions, "profiles": hot.profiles, "units": hot.units}, set())  # fmt: skip
    assert not _former(found) and {"ASC", "ASC (Sdc1+)"} <= set(hot.module_maps)
    rep = spc.run(spc.datasets.synthetic_example(), n_perm=19, rng=0)
    texts = _strings({"tests": rep.tests(), "params": rep.params, "methods": rep.methods()}, set())
    assert not _former(texts)
    assert rep.params["cell_types"]["own_types"]["BCR"]["labels"] == ["ASC", "B", "GC B"]


@pytest.mark.parametrize("spelling", ["plasma_cell", "pc", "plasma", "Asc"])
def test_region_kind_spelling_is_canonical(spelling):
    from spaclone.gex.regions import _kind

    assert _kind(spelling) == "asc" and _kind("asc") == "asc"
    assert np.array_equal(gex.REGION_MARKERS[_kind(spelling)]["human"], ["JCHAIN", "XBP1", "PRDM1", "SDC1"])
