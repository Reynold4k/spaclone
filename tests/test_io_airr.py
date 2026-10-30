from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spaclone import io, validate_air
from spaclone._utils import SpacloneWarning
from spaclone.io import airr as A
from spaclone.pp import define_clones, filter_valid

DATA = Path(__file__).resolve().parent / "_data"
BCR, TCR = DATA / "airr_stereoir_bcr.tsv", DATA / "airr_stereoir_tcr.tsv"
CELLS = {"B lineage": DATA / "cells_b.tsv", "T": DATA / "cells_t.tsv"}


def _stereoir(**kw):
    return io.read_airr([BCR, TCR], cell_types=CELLS, library_id="mini", technology="StereoIR", **kw)


def test_stereoir_contigs_spots_and_units():
    sa = _stereoir()
    air = sa.air
    validate_air(air)
    assert len(air) == 8
    assert air[air.locus.str.startswith("IG")].spot_id.nunique() == 4
    assert air[air.locus.str.startswith("TR")].spot_id.nunique() == 2
    # cell_id "x-y" in DNB, 0.5 um per DNB, no half-bin shift
    r = air.iloc[0]
    assert (r.spot_id, r.x_um, r.y_um, r.x_src, r.y_src) == ("100-200", 50.0, 100.0, 100.0, 200.0)
    assert sa.meta["um_per_unit"] == 0.5 and "DNB" in sa.meta["source_unit"]
    assert air.attrs["spaclone"]["um_per_unit"] == 0.5
    assert sa.meta["level"] == "contig" and sa.meta["molecule_level"]
    # TRUST4 '*' = no call; only the first of several V calls is kept
    assert air.c_call.isna().sum() == 1 and air.d_call.isna().sum() == 4
    assert air.loc[5, "v_call"] == "IGHV1-26*01"
    assert air.productive.dtype == bool and int((~air.productive).sum()) == 1
    assert air.consensus_count.dtype == np.float64 and air.consensus_count.iloc[2] == 1.0
    assert air.library_id.eq("mini").all()


def test_exact_clone_ids_match_define_clones():
    air = _stereoir().air
    assert air.loc[0, "clone_id"] == "IGH|IGHV2-2|IGHJ3|CARNSHYYGSRYPFAYW"
    # the allele differs (IGHV2-2*01 / *02) but the clone is the same
    assert air.loc[0, "clone_id"] == air.loc[2, "clone_id"]
    m4 = define_clones(air, method="exact", key_added="m4")
    assert (m4.m4.astype(str).to_numpy() == air.clone_id.to_numpy()).all()


def test_spot_table_keeps_typed_spots_without_records():
    sa = _stereoir()
    g = sa.gex
    assert isinstance(g, ad.AnnData) and g.n_vars == 0
    assert list(g.obs_names) == sorted(["100-200", "120-200", "140-220", "160-240", "180-260", "200-300"])
    assert g.obs.loc["180-260", "n_records"] == 0
    assert g.obs.loc["180-260", "cell_type"] == "B lineage"
    assert g.obs.loc["140-220", "cell_type"] == "untyped"
    np.testing.assert_allclose(g[["180-260"]].obsm["spatial"], [[90.0, 130.0]])
    assert sa.air.set_index("spot_id").loc["200-300", "cell_type"] == "T"
    assert sa.meta["n_spots_table"] == 6


def test_records_feed_the_validity_filter():
    ok, rep = filter_valid(_stereoir().air, require_productive=True, min_reads=2)
    # CARDY*YW (stop), the non-productive IGK contig and the two 1-read... contigs fail
    assert rep.records.iloc[0] == 8
    assert len(ok) == 5
    assert set(ok.junction_aa) == {"CARNSHYYGSRYPFAYW", "CQNDYSYPLTF", "CALWYSNQFIF", "CASSLGQNTLYF", "CAMREGSNYQLIW"}


def test_longairr_columns_and_umi_level():
    sa = io.read_airr(DATA / "airr_longairr.tsv", coords=("x", "y"), spot_key="spbcid", scale_um=2.0)
    air = sa.air
    assert len(air) == 4 and air.spot_id.nunique() == 3
    np.testing.assert_allclose(air[["x_um", "y_um"]].to_numpy()[0], [20.0, 40.0])
    assert sa.meta["level"] == "molecule" and sa.meta["has_molecule_id"]
    assert air.umi.iloc[0] == "AACCGGTTAA"
    assert air.d_call.isna().iloc[2]
    assert sa.gex is None
    assert sa.meta["library_id"] == "airr_longairr"


def test_coordinate_columns_without_spot_key_format_ids():
    df = pd.read_csv(DATA / "airr_longairr.tsv", sep="\t").drop(columns="spbcid")
    sa = io.read_airr(_write(df), coords=("x", "y"), scale_um=(2.0, 3.0))
    assert sa.air.spot_id.tolist()[:3] == ["10_20", "10_20", "11_20"]
    np.testing.assert_allclose(sa.air[["x_um", "y_um"]].to_numpy()[0], [20.0, 60.0])
    assert sa.meta["um_per_unit"] == [2.0, 3.0]


def _write(df, name="t.tsv"):
    import tempfile

    p = Path(tempfile.mkdtemp()) / name
    df.to_csv(p, sep="\t", index=False)
    return p


def test_positions_table_keeps_off_tissue_barcodes():
    df = pd.read_csv(BCR, sep="\t").drop(columns=["Unnamed: 0"])
    df["cell_id"] = ["AAAC", "AAAC", "CCCG", "GGGT", "GGGT", "NOPOS"]
    pos = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 9.0], "y": [1.0, 1.0, 2.0, 9.0], "in_tissue": [1, 1, 0, 0]},
        index=["AAAC", "CCCG", "GGGT", "TTTT"],
    )
    sa = io.read_airr(_write(df), coords=pos, scale_um=50.0)
    assert len(sa.air) == 5 and sa.meta["n_records_unpositioned"] == 1
    assert sa.air.in_tissue.tolist() == [True, True, True, False, False]
    g = sa.gex
    assert list(g.obs_names) == ["AAAC", "CCCG", "GGGT", "TTTT"]
    assert g.obs.loc["TTTT", "n_records"] == 0 and not g.obs.loc["TTTT", "in_tissue"]
    np.testing.assert_allclose(g[["TTTT"]].obsm["spatial"], [[450.0, 450.0]])


def test_clone_key_choices():
    df = pd.read_csv(BCR, sep="\t").drop(columns=["Unnamed: 0"])
    full = df.assign(clone_id=[f"c{i % 3}" for i in range(len(df))])
    sa = io.read_airr(_write(full))
    assert sa.air.clone_id.tolist()[:3] == ["c0", "c1", "c2"] and sa.meta["clone_rule"] == "column 'clone_id'"
    ex = io.read_airr(_write(full), clone_key="exact")
    assert (
        ex.air.clone_id.iloc[0].startswith("IGH|IGHV2-2|")
        and (ex.air.clone_id_airr.iloc[:3] == ["c0", "c1", "c2"]).all()
    )
    part = full.copy()
    part.loc[0, "clone_id"] = np.nan
    with pytest.warns(SpacloneWarning, match="clone_id is empty"):
        p = io.read_airr(_write(part))
    assert p.air.clone_id.iloc[0].startswith("IGH|")
    other = df.assign(lineage=list("aabbcc"))
    o = io.read_airr(_write(other), clone_key="lineage", columns=[*A.AIRR_DEFAULT, "lineage"])
    assert o.air.clone_id.tolist() == list("aabbcc")


def test_locus_inferred_from_v_call_when_missing():
    df = pd.read_csv(TCR, sep="\t").drop(columns=["Unnamed: 0", "locus"])
    sa = io.read_airr(_write(df))
    assert sa.air.locus.tolist() == ["TRB", "TRA"]


def test_reader_errors():
    df = pd.read_csv(BCR, sep="\t").drop(columns=["Unnamed: 0"])
    with pytest.raises(ValueError, match=r"cell_id 'bin_7' is not of the form") as e:
        io.read_airr(_write(df.assign(cell_id="bin_7")))
    assert "np.str_" not in str(e.value)
    with pytest.raises(ValueError, match="listed as both"):
        io.read_airr(BCR, cell_types={"a": ["100-200"], "b": ["100-200"]})
    with pytest.raises(ValueError, match="coordinate column"):
        io.read_airr(BCR, coords=("x", "y"))
    with pytest.raises(ValueError, match="more than one coordinate pair"):
        io.read_airr(_write(df.assign(x=np.arange(6), y=0)), coords=("x", "y"), spot_key="cell_id")
    with pytest.raises(ValueError, match="scale_um"):
        io.read_airr(BCR, scale_um=0)
    with pytest.raises(ValueError, match=r"no clone id \(junction_aa is empty"):
        io.read_airr(_write(df.assign(junction_aa=np.nan)))
    with pytest.raises(ValueError, match=r"no clone id \(the table has no junction_aa column.*clone_key="):
        io.read_airr(_write(df.drop(columns=["junction_aa"])))
    with pytest.raises(ValueError, match="one library per call"):
        io.read_airr([BCR, TCR], cell_types=CELLS, library_id=["a", "b"])


def test_unit_scale_invariance_and_h5ad_roundtrip(tmp_path):
    a = io.read_airr(BCR)
    b = io.read_airr(BCR, scale_um=1.0)
    np.testing.assert_allclose(b.air.x_um.to_numpy(), 2 * a.air.x_um.to_numpy())
    pd.testing.assert_frame_equal(a.air.drop(columns=["x_um", "y_um"]), b.air.drop(columns=["x_um", "y_um"]))
    sa = _stereoir()
    sa.gex.write_h5ad(tmp_path / "spots.h5ad")
    back = ad.read_h5ad(tmp_path / "spots.h5ad")
    assert back.uns["spaclone"]["um_per_unit"] == 0.5


def test_helpers():
    assert A.clean_call(["IGHV1-26*00(1570),IGHV1-18*00(1500)", "*", "", None, " IGKC "]).tolist()[0] == "IGHV1-26*00"
    c = A.clean_call(["*", "", None, " IGKC "])
    assert pd.isna(c[:3]).all() and c[3] == "IGKC"
    assert A.gene_of(["TRAV13-4/DV7*01", "IGHV1-26*00(10)"]).tolist() == ["TRAV13-4/DV7", "IGHV1-26"]
    assert A.locus_from_call(["IGKV8-19*01", "XYZ1"]).tolist()[0] == "IGK" and pd.isna(A.locus_from_call(["XYZ1"])[0])
    key = A.exact_clone_ids(["IGH", "IGH"], ["IGHV1*01", np.nan], ["IGHJ1*02", "IGHJ1"], ["CARW", ""])
    assert key[0] == "IGH|IGHV1|IGHJ1|CARW" and pd.isna(key[1])
    assert A.format_spots([1, 20], [3, 4], "{x}-{y}").tolist() == ["1-3", "20-4"]
    with pytest.raises(ValueError):
        A.format_spots([1], [2], "{x:03d}")
    assert A.modal_codes(np.array([0, 0, 0, 1, 1]), np.array([2, 1, 1, 3, 0]), 3).tolist() == [1, 0, -1]
    assert A.to_bool(["T", "f", "TRUE", "0"]).tolist() == [True, False, True, False]
    assert A.to_bool(["T", None]).dtype == "boolean"
    assert A.str_values([1, None, "a"]).tolist()[0] == "1"
    assert pd.isna(A.str_values([1, None, "a"])[1])
