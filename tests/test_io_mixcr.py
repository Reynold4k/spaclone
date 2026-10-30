import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pytest

from spaclone import io, validate_air
from spaclone._utils import SpacloneWarning

DATA = Path(__file__).resolve().parent / "_data"
V12, PUB = DATA / "xcr_meta_v12.csv", DATA / "xcr_public_meta.csv"
BCR = ("IGH", "IGK", "IGL")


def _parquet(src, dst):
    t = pacsv.read_csv(
        src,
        convert_options=pacsv.ConvertOptions(column_types={"x": pa.int64(), "y": pa.int64(), "cloneId": pa.string()}),
    )
    pq.write_table(t, dst, row_group_size=3)
    return dst


def test_v12_units_reads_and_no_molecule_id():
    sa = io.read_stereo_xcr(V12)
    air = sa.air
    validate_air(air)
    m = sa.meta
    assert m["layout"] == "mixcr_meta" and m["level"] == "spot_clone"
    assert m["umi_equals_cid_fraction"] == 1.0 and not m["has_molecule_id"]
    assert m["n_rows_read"] == 10 and m["n_reads"] == 9 and m["n_reads_unassigned"] == 1
    assert m["reads_by_locus"] == {"IGH": 3, "IGK": 4, "TRB": 2}
    u = air.set_index(["spot_id", "clone_id"])
    assert len(air) == 5
    assert u.loc[("100_200", "0"), "consensus_count"] == 3 and u.loc[("100_200", "0"), "n_cid"] == 2
    assert u.loc[("100_200", "1"), "f_functional"] == 0.5
    # an off-tissue DNB far from everything is kept, with exact DNB -> um conversion
    far = u.loc[("5000_5000", "1")]
    assert (far.x_um, far.y_um, far.x_src) == (2500.0, 2500.0, 5000)
    # clone-level AIRR fields from exportClones
    assert u.loc[("100_200", "1"), "junction_aa"] == "CARGITVVANPDYW" and u.loc[("100_200", "1"), "c_call"] == "IGHM"
    assert air.productive.dtype == bool and air.umi.isna().all()
    assert air.consensus_count.sum() == 9


def test_loci_filter_and_reads_mode():
    sa = io.read_stereo_xcr(V12, loci=BCR)
    assert len(sa.air) == 4 and sa.meta["n_reads"] == 7 and set(sa.air.locus) == {"IGH", "IGK"}
    r = io.read_stereo_xcr(V12, collapse=None, loci=BCR)
    assert len(r.air) == 7 and r.meta["level"] == "read" and (r.air.consensus_count == 1).all()
    # read-level CDR3 and functional flag, clone-level junction
    nf = r.air[~r.air.functional]
    assert len(nf) == 1 and nf.cdr3_aa.iloc[0] == "CARGITVV*NPDYW" and nf.junction_aa.iloc[0] == "CARGITVVANPDYW"
    assert r.air.cid.str.len().eq(25).all()


def test_molecule_requested_without_mid_falls_back_with_warning():
    with pytest.warns(SpacloneWarning, match="no molecule identifier"):
        sa = io.read_stereo_xcr(V12, collapse="molecule")
    assert sa.meta["collapse"] == "spot_clone" and len(sa.air) == 5
    assert io.read_stereo_xcr(V12, collapse="auto").meta["collapse"] == "spot_clone"


def test_parquet_gzip_and_batches_agree(tmp_path):
    ref = io.read_stereo_xcr(V12)
    pqt = io.read_stereo_xcr(_parquet(V12, tmp_path / "m.parquet"), library_id="xcr_meta_v12", batch_rows=2)
    pd.testing.assert_frame_equal(ref.air, pqt.air)
    with open(V12, "rb") as fi, gzip.open(tmp_path / "m.csv.gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    gz = io.read_stereo_xcr(tmp_path / "m.csv.gz", library_id="xcr_meta_v12")
    pd.testing.assert_frame_equal(ref.air, gz.air)


def test_extra_columns_are_clone_level_or_read_level():
    sa = io.read_stereo_xcr(V12, columns=["duplicate_count"])
    assert sa.air.set_index("clone_id").duplicate_count.astype(float).groupby(level=0).first().to_dict() == {
        "0": 4.0,
        "1": 3.0,
        "2": 2.0,
    }
    r = io.read_stereo_xcr(V12, collapse=None, columns=["readId"])
    assert r.air.readId.is_unique
    with pytest.raises(ValueError, match="not in the meta table"):
        io.read_stereo_xcr(V12, columns=["nope"])


def test_public_layout_molecules_with_mid():
    sa = io.read_stereo_xcr(PUB, collapse="auto")
    m = sa.meta
    assert m["layout"] == "stereo_xcr_public" and m["has_molecule_id"] and m["collapse"] == "molecule"
    assert m["umi_equals_cid_fraction"] == 0.0
    air = sa.air.set_index(["spot_id", "umi"])
    # a molecule whose reads disagree takes its most frequent clone
    mol = air.loc[("10_20", "TTTTTTTTTT")]
    assert (mol.clone_id, mol.consensus_count, mol.n_clones_in_molecule) == ("SR_0", 3.0, 2)
    assert len(sa.air) == 4
    # MiXCR hit scores are stripped, the first hit is kept
    assert mol.v_call == "IGKV8-19*00"
    assert not air.loc[("30_40", "AAAAAAAAAA"), "productive"]


def test_public_layout_spot_clone_counts_mids():
    sa = io.read_stereo_xcr(PUB, collapse="spot_clone")
    u = sa.air.set_index(["spot_id", "clone_id"])
    assert u.loc[("10_20", "SR_0"), "n_umi"] == 2 and u.loc[("10_20", "SR_0"), "consensus_count"] == 3
    assert set(sa.air.clone_id) == {"SR_0", "SR_5", "SR_1", "LR_IGH_7"}


def test_vectorised_units_match_pandas_groupby(tmp_path):
    rng = np.random.default_rng(3)
    n = 3000
    clones = rng.integers(0, 40, n)
    chain = np.where(clones < 25, "IGK", "IGH")
    df = pd.DataFrame(
        {
            "readId": [f"r{i}" for i in range(n)],
            "topChains": chain,
            "cloneId": clones.astype(str),
            "x": rng.integers(0, 30, n),
            "y": rng.integers(0, 30, n),
            "cid": rng.choice(["A" * 25, "C" * 25, "G" * 25], n),
            "functional": rng.choice(["Functional", "NonFunctional"], n, p=[0.9, 0.1]),
            "junction_aa": "CARW",
        }
    )
    df["umi"] = df["cid"]
    p = tmp_path / "rand.csv"
    df.to_csv(p, index=False)
    sa = io.read_stereo_xcr(p, batch_rows=257)
    g = df.groupby(["x", "y", "topChains", "cloneId"])
    ref = pd.DataFrame(
        {
            "reads": g.size(),
            "n_cid": g.cid.nunique(),
            "f_functional": g.functional.apply(lambda s: (s == "Functional").mean()),
        }
    ).reset_index()
    ref["spot_id"] = ref.x.astype(str) + "_" + ref.y.astype(str)
    got = sa.air.merge(ref, left_on=["spot_id", "locus", "clone_id"], right_on=["spot_id", "topChains", "cloneId"])
    assert len(got) == len(ref) == len(sa.air)
    np.testing.assert_array_equal(got.consensus_count.to_numpy(), got.reads.to_numpy())
    np.testing.assert_array_equal(got.n_cid_x.to_numpy(), got.n_cid_y.to_numpy())
    np.testing.assert_allclose(got.f_functional_x.to_numpy(), got.f_functional_y.to_numpy())


def test_errors(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="unrecognised"):
        io.read_stereo_xcr(bad)
    with pytest.raises(ValueError, match="collapse"):
        io.read_stereo_xcr(V12, collapse="cells")
    with pytest.raises(ValueError, match="spot_format"):
        io.read_stereo_xcr(V12, spot_format="{x:05d}")


def test_most_frequent_read_cdr3_next_to_the_consensus(tmp_path):
    df = pd.read_csv(V12, dtype=str)
    # clone 1: two reads now carry a variant CDR3, one the consensus -> the top read CDR3 is the variant
    m = df.cloneId == "1"
    df.loc[m, "aaSeqCDR3"] = ["CARGITVVENPDYW", "CARGITVVENPDYW", "CARGITVVANPDYW"]
    p = tmp_path / "variant.csv"
    df.to_csv(p, index=False)
    sa = io.read_stereo_xcr(p)
    u = sa.air.set_index("clone_id")
    assert (u.loc["1", "junction_aa"] == "CARGITVVANPDYW").all()
    assert (u.loc["1", "junction_aa_top"] == "CARGITVVENPDYW").all()
    assert (u.loc["0", "junction_aa_top"] == "CQQYSSYPLTF").all()
    assert "junction_aa_top" not in io.read_stereo_xcr(PUB).air.columns


def test_top_cdr3_ties_go_to_the_first_read(tmp_path):
    df = pd.read_csv(V12, dtype=str)
    m = df.cloneId == "1"
    df.loc[m, "aaSeqCDR3"] = ["CARGITVVKNPDYW", "CARGITVVENPDYW", "CARGITVVANPDYW"]  # three variants, one read each
    p = tmp_path / "tie.csv"
    df.to_csv(p, index=False)
    sa = io.read_stereo_xcr(p, batch_rows=2)
    assert (sa.air.set_index("clone_id").loc["1", "junction_aa_top"] == "CARGITVVKNPDYW").all()
    ref = df[m].aaSeqCDR3.value_counts().index[0]  # the archive's rule
    assert ref == "CARGITVVKNPDYW"


def test_parse_mixcr_alignment_known_answers():
    from spaclone.io import MIXCR_ALIGNMENT_FIELDS, parse_mixcr_alignment

    v = pd.Series(
        [
            "223|373|391|0|150|SA223G|1470.0",  # one substitution
            "0|366|387|34|399|DG16DA17SA199GSC231TI329TSG357T|3415.0",  # deletions and an insertion between substitutions
            "12|372|391|0|358|SA12GDC43DT44|3477.0",
            "",
            None,
            "345|372|393|0|27||270.0;342|369|390|0|27|SA350C|270.0",  # two hits
            "10|110|300|0|100|SA20NSC50T|100.0",  # an ambiguous substitution
        ],
        index=list("abcdefg"),
    )
    r = parse_mixcr_alignment(v)
    assert list(r.columns) == list(MIXCR_ALIGNMENT_FIELDS) and list(r.index) == list("abcdefg")
    assert r.substitutions.tolist()[:3] == [1, 3, 1] and r.deletions.tolist()[:3] == [0, 2, 2]
    assert r.insertions.tolist()[:3] == [0, 1, 0]
    assert r.aligned_nt.tolist()[:3] == [150, 364, 358]  # span - deletions
    assert r.loc["d"].isna().iloc[:11].all() and r.loc["e"].isna().iloc[:11].all()
    assert r.loc["f", "substitutions"] == 0 and parse_mixcr_alignment(v, hit=1).loc["f", "substitutions"] == 1
    assert (r.loc["g", "substitutions"], r.loc["g", "ambiguous_substitutions"], r.loc["g", "aligned_nt"]) == (1, 1, 99)
    lax = parse_mixcr_alignment(v, skip_ambiguous=False)
    assert (lax.loc["g", "substitutions"], lax.loc["g", "aligned_nt"]) == (2, 100)
    # a 5' trim counts sequenced (read) positions: two deletions before position 22 shift the read by -2
    t = parse_mixcr_alignment(v, trim_5prime_nt=[0, 20, 5, 0, 0, 0, 15])
    assert t.loc["b", "aligned_nt"] == 344 and t.loc["b", "substitutions"] == 3
    assert t.loc["c", "substitutions"] == 0 and t.loc["c", "aligned_nt"] == 353
    assert (t.loc["g", "substitutions"], t.loc["g", "ambiguous_substitutions"], t.loc["g", "aligned_nt"]) == (1, 0, 85)
    with pytest.raises(ValueError, match="trim"):
        parse_mixcr_alignment(v, trim_5prime_nt=[1, 2])
