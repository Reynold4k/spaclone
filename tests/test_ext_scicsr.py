"""Tests of spaclone.ext.scicsr: the R data reader, signatures, export to sciCSR, agreement metrics, the R bridge."""

import os
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import bcr
from spaclone.ext import scicsr

DATA = Path(__file__).parent / "_data"


def test_read_rdata_matrix_gzip_and_bzip2():
    for f in ("s1_toy_nmf.rda", "s1_toy_nmf_bz.rda"):
        d = scicsr.read_rdata(DATA / f)
        m = d["toy_nmf"]
        assert list(m.index) == ["Ighm-S", "Ighm-P", "Igha-S", "Igha-P"]
        np.testing.assert_array_equal(m.to_numpy(), [[2.5, 0], [1.25, 0.5], [0, 1.75], [0.125, 0.375]])


def test_read_rdata_rds_vectors_lists_altrep():
    x = scicsr.read_rdata(DATA / "s1_toy_list.rds")  # xz-compressed, R >= 3.5 (ALTREP compact sequence)
    assert x["a"].index.tolist() == ["u", "v", "w"] and np.isnan(x["a"]["v"]) and x["a"]["w"] == 3
    assert x["b"].tolist() == ["x", None, "z"]
    assert x["c"][0] == 1 and np.isnan(x["c"][2])
    np.testing.assert_array_equal(x["d"], np.arange(1, 7))
    assert x["e"].loc["r2", "c3"] == 3.0
    with pytest.raises(ValueError):
        scicsr.read_rdata(b"not R data")


def test_nmf_signatures_from_file(tmp_path):
    s = scicsr.nmf_signatures("mouse", path=DATA / "s1_toy_nmf.rda")
    assert list(s.columns) == ["naive", "switched"] and s.loc["Ighm-S", "naive"] == 2.5
    csv = tmp_path / "sig.csv"
    s.to_csv(csv)
    pd.testing.assert_frame_equal(scicsr.nmf_signatures("mouse", path=csv), s, check_names=False)
    # the toy signature drives the reference-based potential
    st = pd.DataFrame({"IgM_S": [9, 0], "IgM_P": [5, 0], "IgA_S": [0, 6], "IgA_P": [0, 9]}, index=["a", "b"])
    units = pd.DataFrame({"c_call": ["IGHM", "IGHA"], "x_um": [0.0, 1.0], "y_um": [0.0, 0.0]}, index=["a", "b"])
    p = bcr.csr_potential(units, unit_key=None, sterile=st, reference=s, impute=None, library_size=None)
    assert p.loc["a", "csr_pot"] == 0.0 and p.loc["b", "csr_pot"] == 1.0


def test_scicsr_isotype_labels():
    assert scicsr.scicsr_isotype_labels(["IgM", "IgG2b", "IgA1", None, "unknown"]) == ["M", "G2b", "A1", None, None]


def test_to_scicsr_files(tmp_path):
    units = pd.DataFrame(
        {
            "isotype": ["IgM", "IgG1", "IgA"],
            "csr_pot": [0.0, 0.4, 1.0],
            "shm": [0.1, 0.5, 0.9],
            "x_um": [0.0, 10.0, 20.0],
            "y_um": [0.0, 0.0, 5.0],
            "region": ["F", "GC", "RP"],
        },
        index=["u1", "u2", "u3"],
    )
    g = sparse.csr_matrix(np.array([[0, 1, 0], [1, 0, 0.5], [0, 0.5, 0]]))
    st = pd.DataFrame({"IgM_S": [1, 0, 0], "IgM_P": [2, 0, 0]}, index=["u1", "u2", "u3"])
    gex = ad.AnnData(X=sparse.csr_matrix(np.eye(3)), obs=pd.DataFrame(index=["u1", "u2", "u3"]))
    paths = scicsr.to_scicsr(units, gex, path=tmp_path / "x", graph=g, sterile=st, columns=["region"])
    a = ad.read_h5ad(paths["h5ad"])
    assert a.obs["isotype"].tolist() == ["M", "G1", "A"] and a.obs["c_gene"].tolist() == ["IGHM", "IGHG1", "IGHA"]
    assert a.obs["region"].tolist() == ["F", "GC", "RP"]
    np.testing.assert_array_equal(a.obsp["connectivities"].toarray(), g.toarray())
    np.testing.assert_array_equal(a.obsm["spatial"][:, 0], [0, 10, 20])
    ig = pd.read_csv(paths["ighc"], index_col=0)
    assert list(ig.columns) == ["Ighm-S", "Ighm-P"]
    assert {"meta", "rna", "genes", "graph"} <= set(paths)


def test_compare_identical_and_different():
    s = pd.Series([0.1, 0.5, 0.9], index=["a", "b", "c"])
    t = sparse.random(5, 5, density=0.5, random_state=0, format="csr")
    f = pd.DataFrame(np.arange(9.0).reshape(3, 3), index=list("xyz"), columns=list("xyz"))
    nat = {
        "csr_pot": s,
        "isotype": pd.Series(["M", "G1", "A"], index=s.index),
        "transition_matrix": t,
        "gross_flux": f,
        "mfpt": 10.0,
    }
    res = scicsr.compare(nat, nat).set_index(["quantity", "metric"]).value
    assert res[("csr_pot", "max_abs_diff")] == 0 and res[("csr_pot", "pearson")] == pytest.approx(1.0)
    assert res[("isotype", "agreement")] == 1.0 and res[("transition_matrix", "rel_frobenius")] == 0
    assert res[("mfpt", "rel_diff")] == 0
    other = {**nat, "csr_pot": s + 0.01, "mfpt": 11.0}
    r2 = scicsr.compare(nat, other).set_index(["quantity", "metric"]).value
    assert r2[("csr_pot", "max_abs_diff")] == pytest.approx(0.01) and r2[("mfpt", "rel_diff")] == pytest.approx(1 / 11)


@pytest.mark.slow
def test_run_r_against_installed_scicsr():
    """Run the original sciCSR on a toy units table (needs SPACLONE_SCICSR_RSCRIPT pointing to an R with sciCSR)."""
    rscript = os.environ.get("SPACLONE_SCICSR_RSCRIPT")
    if not rscript or not shutil.which(rscript):
        pytest.skip("set SPACLONE_SCICSR_RSCRIPT to an Rscript with sciCSR installed")
    rng = np.random.default_rng(0)
    n = 60
    iso = rng.choice(["IGHM", "IGHG1", "IGHG2B", "IGHA", None], n)
    units = pd.DataFrame(
        {
            "c_call": iso,
            "v_identity": rng.uniform(90, 100, n),
            "x_um": rng.uniform(0, 300, n),
            "y_um": rng.uniform(0, 300, n),
        },
        index=[f"c{i}" for i in range(n)],
    )
    pot = bcr.csr_potential(units, unit_key=None, impute="spatial", k=5, class_only="missing")
    adj = bcr.csr_dynamics._knn_adjacency(pot[["x_um", "y_um"]].to_numpy(), 5, None)
    observed = pot.isotype.where(~pot.imputed, None)  # sciCSR imputes the same units on the same graph
    tab = pot.assign(isotype=observed, v_identity=units.v_identity)
    out = scicsr.run_r(tab, rscript=rscript, graph=adj, steps=("csr_potential", "shm"))
    lab = pot.isotype.map(lambda v: scicsr.scicsr_isotype_labels([v])[0])
    assert (out["meta"].isotype.loc[pot.index] == lab).all()
    np.testing.assert_allclose(out["meta"].csr_pot.loc[pot.index], pot.csr_pot, atol=1e-10)
