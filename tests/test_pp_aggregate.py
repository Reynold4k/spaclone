import numpy as np
import pandas as pd
import pytest

from spaclone.pp import CellAssignmentWarning, aggregate, assign_to_cells


def _toy():
    return pd.DataFrame(
        {
            "spot_id": ["a", "a", "b", "c", "c", "d", "e", "e"],
            "x_um": [0.0, 0, 10, 20, 20, 30, 40, 40],
            "y_um": [0.0] * 8,
            "locus": ["IGH", "IGK", "IGH", "TRA", "TRB", "IGH", "IGK", "IGK"],
            "clone_id": ["h1", "k1", "h1", "a1", "b1", "h2", "k1", "k2"],
            "junction_aa": ["CARDYW", "CQQYF", "CARDYW", "CAVRW", "CASSF", "CARDYW", "CQQYF", "CQQLF"],
            "consensus_count": [5, 1, 9, 3, 2, np.nan, 2, 7],
            "in_tissue": [True, True, True, True, False, True, True, True],
        }
    )


def test_aggregate_spot_by_clone_counts():
    ad = aggregate(_toy())
    assert ad.shape == (5, 6)
    X = pd.DataFrame(ad.X.toarray(), index=ad.obs_names, columns=ad.var_names)
    assert X.loc["e", "k1"] == 1 and X.loc["e", "k2"] == 1 and X.loc["a", "h1"] == 1 and X.values.sum() == 8
    assert ad.obs.loc["a", "n_records"] == 2 and ad.obs.loc["a", "n_loci"] == 2 and ad.obs.loc["c", "n_clones"] == 2
    assert ad.var.loc["h1", "locus"] == "IGH" and ad.var.loc["k1", "n_units"] == 2 and ad.var.loc["h1", "n_reads"] == 14
    np.testing.assert_allclose(ad.obsm["spatial"][:, 0], [0, 10, 20, 30, 40])
    assert ad.obs.loc["c", "in_tissue"] is not None
    r = aggregate(_toy(), value="reads")
    assert pd.DataFrame(r.X.toarray(), index=r.obs_names, columns=r.var_names).loc["e", "k2"] == 7


def test_aggregate_bins_min_records_and_libraries():
    ad = aggregate(_toy(), bin_um=25)
    assert ad.n_obs == 2 and ad.obs.n_records.tolist() == [5, 3]
    np.testing.assert_allclose(ad.obsm["spatial"], [[12.5, 12.5], [37.5, 12.5]])
    assert aggregate(_toy(), min_records=2).n_obs == 3
    two = pd.concat([_toy().assign(library_id="L1"), _toy().assign(library_id="L2")], ignore_index=True)
    ad2 = aggregate(two)
    assert ad2.n_obs == 10 and "L1:a" in ad2.obs_names and ad2.X.sum() == 16
    # row order does not matter
    ad3 = aggregate(two.sample(frac=1, random_state=0))
    a = pd.DataFrame(ad2.X.toarray(), index=ad2.obs_names, columns=ad2.var_names)
    b = pd.DataFrame(ad3.X.toarray(), index=ad3.obs_names, columns=ad3.var_names).loc[a.index, a.columns]
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError):
        aggregate(_toy(), bin_um=0)
    with pytest.raises(KeyError):
        aggregate(_toy(), unit="cell")


def test_assign_to_cells_polygons_and_label_image():
    air = _toy()
    cells = {"c1": np.array([[-5, -5], [15, -5], [15, 5], [-5, 5]]), "c2": np.array([[35, -5], [45, -5], [45, 5]])}
    with pytest.warns(CellAssignmentWarning):
        res = assign_to_cells(air, cells)
    assert res.air.cell_id.tolist() == ["c1", "c1", "c1", "c2", "c2"]
    rep = res.report.set_index("library")
    assert rep.loc["all", "records_in_cells"] == 5 and rep.loc["all", "records_outside"] == 3
    assert rep.loc["all", "cells_with_records"] == 2
    keep = assign_to_cells(air, cells, drop_outside=False, warn=False).air
    assert len(keep) == 8 and keep.cell_id.isna().sum() == 3
    lab = np.zeros((2, 6), int)
    lab[:, 0:2] = 1
    lab[:, 4:6] = 7
    img = assign_to_cells(air, lab, pixel_um=10, origin_um=(-5, -10), warn=False).air
    assert img.cell_id.tolist() == ["1", "1", "1", "7", "7"]
    with pytest.raises(ValueError):
        assign_to_cells(air, lab, warn=False)
    # one segmentation per library
    two = pd.concat([air.assign(library_id="L1"), air.assign(library_id="L2")], ignore_index=True)
    per = assign_to_cells(
        two, {"L1": cells, "L2": {"z": np.array([[15, -5], [25, -5], [25, 5], [15, 5]])}}, per_library=True, warn=False
    )
    assert set(per.air.cell_id[per.air.library_id == "L2"]) == {"z"}
    assert set(per.report.library) == {"L1", "L2", "all"}
    agg = aggregate(per.air, unit="cell")
    assert agg.n_obs == 3
