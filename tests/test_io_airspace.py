from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from spaclone import io, validate_air
from spaclone.io.airspace import tissue_beads

DATA = Path(__file__).resolve().parent / "_data"
CT = DATA / "airspace_mini_ovp_clone_table.txt"


def _beads():
    """36 beads on a 6 x 6 grid (10 units apart, 150 counts: tissue), one low-count bead, one isolated bead."""
    g = np.arange(6) * 10.0
    xx, yy = np.meshgrid(g, g)
    bc = [f"BEAD{i:03d}" for i in range(36)] + ["BEAD098", "BEAD099"]
    x, y = np.r_[xx.ravel(), 25.0, 5000.0], np.r_[yy.ravel(), 25.0, 5000.0]
    counts = np.full((38, 3), 50.0, np.float32)
    counts[36] = 10
    obs = pd.DataFrame({"barcode": bc, "x": x, "y": y}, index=bc)
    return ad.AnnData(X=sp.csc_matrix(counts), obs=obs, var=pd.DataFrame(index=["Igkc", "Ighm", "Actb"]))


def test_molecules_modal_representative_and_tissue():
    sa = io.read_airspace(_beads(), CT)
    air = sa.air
    validate_air(air)
    assert len(air) == 6 and sa.meta["n_records_unmatched_barcode"] == 1
    mol = air.set_index(["spot_id", "umi", "locus"]).loc[("BEAD000", "AAAAAAA", "IGK")]
    # 3 reads: two of clone 10 (CQQYSSYPLTF) after one of clone 11 -> modal CDR3 wins
    assert (mol.clone_id, mol.junction_aa, mol.consensus_count, mol.n_clones_in_molecule) == ("10", "CQQYSSYPLTF", 3, 2)
    assert mol.v_call == "IGKV8-19*00" and mol.c_call == "IGKC"
    assert air.productive.tolist() == [True, True, True, False, True, True]
    # 0.65 um per Curio unit; the isolated bead is kept, off the tissue
    far = air[air.spot_id == "BEAD099"].iloc[0]
    assert far.x_um == pytest.approx(3250.0) and not far.in_tissue
    assert air[air.spot_id != "BEAD099"].in_tissue.all()
    assert sa.meta["n_beads"] == 38 and sa.meta["n_tissue_beads"] == 36 and sa.meta["um_per_unit"] == 0.65
    assert sa.meta["library_id"] == "airspace_mini"


def test_first_read_representative_matches_drop_duplicates():
    sa = io.read_airspace(_beads(), CT, representative="first", loci=["IGH", "IGK", "IGL"])
    M = pd.read_csv(CT, sep="\t", dtype=str)
    M = M[M.receptor_type.isin(["IGH", "IGK", "IGL"])].drop_duplicates(["barcode", "umi", "receptor_type"])
    M = M[M.barcode.str.startswith("BEAD")]
    assert sa.air.clone_id.tolist() == M.cloneId.tolist()
    assert sa.air.loc[0, "clone_id"] == "11" and len(sa.air) == 5


def test_gex_holds_every_bead():
    sa = io.read_airspace(_beads(), CT)
    g = sa.gex
    assert g.n_obs == 38 and g.n_vars == 3
    assert g.obs.in_tissue.sum() == 36 and not g.obs.loc["BEAD098", "in_tissue"]
    np.testing.assert_allclose(g.obs.total_counts.to_numpy()[:2], [150.0, 150.0])
    np.testing.assert_allclose(g[["BEAD001"]].obsm["spatial"], [[6.5, 0.0]])
    assert g.obs.library_id.eq("airspace_mini").all()
    lite = io.read_airspace(_beads(), CT, keep_x=False).gex
    assert lite.n_vars == 0 and lite.obs.in_tissue.sum() == 36


def test_reads_mode_and_h5ad_path(tmp_path):
    p = tmp_path / "GSM0000001_adata_D99_with_image_raw_counts.h5ad"
    _beads().write_h5ad(p)
    ct = tmp_path / "GSM0000002_D99_ovp_clone_table.txt"
    ct.write_text(CT.read_text())
    sa = io.read_airspace(p, ct, collapse=None)
    assert len(sa.air) == 8 and (sa.air.consensus_count == 1).all() and "read_id" in sa.air
    assert sa.meta["library_id"] == "D99" and sa.meta["level"] == "read"


def test_tissue_rule_counts_neighbours_within_radius():
    xy = np.array([[0, 0], [1, 0], [2, 0], [10, 0]], float)
    tot = np.array([200, 200, 200, 200])
    assert tissue_beads(xy, tot, min_neighbours=2, radius=2.0).tolist() == [True, True, True, False]
    assert tissue_beads(xy, tot, min_neighbours=1, radius=1.0).tolist() == [True, True, True, False]
    assert tissue_beads(xy, np.array([200, 50, 200, 200]), min_neighbours=1, radius=2.0).tolist() == [
        True,
        False,
        True,
        False,
    ]


def test_matches_pandas_groupby_on_random_table(tmp_path):
    rng = np.random.default_rng(7)
    n = 2000
    beads = _beads()
    M = pd.DataFrame(
        {
            "readId": np.arange(n).astype(str),
            "aaSeqCDR3": rng.choice(["CQQYSSYPLTF", "CQQYSSYPLTA", "CARDYW"], n),
            "cloneId": rng.integers(0, 5, n).astype(str),
            "umi": rng.choice(["AAAAAAA", "CCCCCCC", "GGGGGGG"], n),
            "barcode": rng.choice(beads.obs_names[:36], n),
            "receptor_type": rng.choice(["IGK", "IGH"], n),
        }
    )
    p = tmp_path / "rand_ovp_clone_table.txt"
    M.to_csv(p, sep="\t", index=False)
    sa = io.read_airspace(beads, p)
    g = M.groupby(["barcode", "umi", "receptor_type"])
    ref = g.agg(reads=("readId", "size"), cdr3=("aaSeqCDR3", lambda v: v.mode().iat[0])).reset_index()
    got = sa.air.merge(ref, left_on=["spot_id", "umi", "locus"], right_on=["barcode", "umi", "receptor_type"])
    assert len(got) == len(ref) == len(sa.air)
    assert (got.consensus_count.to_numpy() == got.reads.to_numpy()).all()
    assert (got.junction_aa.to_numpy() == got.cdr3.to_numpy()).all()


def test_errors():
    with pytest.raises(ValueError, match="collapse"):
        io.read_airspace(_beads(), CT, collapse="bead")
    b = _beads()
    b.obs["barcode"] = "X"
    with pytest.raises(ValueError, match="unique"):
        io.read_airspace(b, CT)


def test_alignments_are_kept_for_the_representative_read(tmp_path):
    M = pd.read_csv(CT, sep="\t", dtype=str, keep_default_na=False)
    M["allVAlignments"] = [f"100|{250 + i}|391|0|{150 + i}|SA{120 + i}G|900.0" for i in range(len(M))]
    M["allJAlignments"] = "0|40|50|150|190||400.0"
    path = tmp_path / "aln_ovp_clone_table.txt"
    M.to_csv(path, sep="\t", index=False)
    plain = io.read_airspace(_beads(), path, alignments=False)
    assert "v_alignment_mixcr" not in plain.air.columns
    sa = io.read_airspace(_beads(), path)  # default "auto": kept when the table has them
    pd.testing.assert_frame_equal(sa.air, io.read_airspace(_beads(), path, alignments=True).air)
    pd.testing.assert_frame_equal(io.read(_beads(), path).air, sa.air)  # spaclone.read keeps them too
    air = sa.air
    assert sa.meta["alignments"] == ["v_alignment_mixcr", "j_alignment_mixcr"]
    mol = air.set_index(["spot_id", "umi", "locus"]).loc[("BEAD000", "AAAAAAA", "IGK")]
    rep = M[(M.barcode == "BEAD000") & (M.umi == "AAAAAAA") & (M.cloneId == "10")].iloc[0]  # modal-CDR3 read
    assert mol.v_alignment_mixcr == rep.allVAlignments and mol.j_alignment_mixcr == "0|40|50|150|190||400.0"
    with pytest.raises(ValueError, match="allVAlignments"):
        io.read_airspace(_beads(), CT, alignments=True)
    assert io.read_airspace(_beads(), CT).meta["alignments"] == []  # "auto" on a table without them
    with pytest.raises(ValueError, match="alignments must be"):
        io.read_airspace(_beads(), CT, alignments="yes")


def test_read_dispatch_honours_unit_overlap_and_unmatched(caplog, capsys):
    from spaclone.io.paired import PairedReadWarning

    ref = io.read_airspace(_beads(), CT)
    with caplog.at_level("WARNING", logger="spaclone"):
        sa = io.read(_beads(), CT, verbose=True)
    pd.testing.assert_frame_equal(sa.air, ref.air)
    assert any(r.levelname == "WARNING" and "not a bead" in r.getMessage() for r in caplog.records)
    m = sa.meta
    assert (m["spot_overlap"], m["n_records_dropped"], m["unmatched_rule"]) == (0.8, 1, "drop")  # 4 of 5 barcodes
    assert m["io"]["receptors"]["format"] == "airspace_ovp" and m["io"]["barcodes"]["overlap_after"] == 0.8
    assert "receptors: airspace_ovp" in capsys.readouterr().out  # verbose prints the decisions
    big = io.read(_beads(), CT, unit_um=1.3)  # unit_um is passed on as the bead scale
    np.testing.assert_allclose(big.air.x_um, big.air.x_src * 1.3)
    assert big.meta["um_per_unit"] == 1.3 and big.meta["unit_rule"] == "given: unit_um = 1.3"
    with pytest.raises(ValueError, match="min_overlap = 90%"):
        io.read(_beads(), CT, min_overlap=0.9)
    with pytest.raises(ValueError, match=r"not beads of the h5ad \(e\.g\. \['NOTABEAD'\]\)"):
        io.read(_beads(), CT, unmatched="error")
    with pytest.raises(ValueError, match="'coordinates'"):
        io.read(_beads(), CT, unmatched="coordinates")
    with pytest.warns(PairedReadWarning, match="does not use spot_key='barcode', bin_size=20"):
        io.read(_beads(), CT, spot_key="barcode", bin_size=20)
    with pytest.warns(PairedReadWarning, match="check scale_um .unit_um in spaclone.read."):
        io.read(_beads(), CT, unit_um=650)  # nm given instead of µm per bead unit


def test_clone_table_of_another_sample_raises(tmp_path):
    """The D14 beads with the D21 clone table: almost no clone-table barcode is a bead."""
    M = pd.read_csv(CT, sep="\t", dtype=str, keep_default_na=False)
    M["barcode"] = [f"D21BEAD{i:03d}" for i in range(len(M) - 1)] + ["BEAD000"]
    other = tmp_path / "GSM0000003_D21_ovp_clone_table.txt"
    M.to_csv(other, sep="\t", index=False)
    with pytest.raises(ValueError, match=r"only 11\.1% of the 9 bead barcodes .* different samples") as e:
        io.read(_beads(), other)
    assert "np.str_" not in str(e.value) and "'D21BEAD000'" in str(e.value)
    assert len(io.read_airspace(_beads(), other, min_overlap=0.0).air) == 1  # the check can be lowered explicitly
