import matplotlib

matplotlib.use("Agg")

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spaclone import bcr, pl, tl

#: Folder with ``ncbi-igblast-1.22.0/``, ``database/`` and ``imgt_mouse/IGHV.fasta`` (the IgBLAST test is skipped
#: when the variable is unset or the folder is missing).
IGBLAST = Path(os.environ.get("SPACLONE_TEST_IGBLAST", "") or "/nonexistent-igblast")
_PLAIN = (type(None), bool, int, float, str, np.generic)


def _plain(value, where="attrs"):
    """Assert that ``value`` holds only scalars and plain containers (no DataFrame, Series or array)."""
    if isinstance(value, dict):
        for k, v in value.items():
            _plain(v, f"{where}[{k!r}]")
    elif isinstance(value, list | tuple | set | frozenset):
        for k, v in enumerate(value):
            _plain(v, f"{where}[{k}]")
    else:
        assert isinstance(value, _PLAIN), f"{where} holds a {type(value).__name__}"


def test_count_mismatches_known_answers():
    q = pd.Series(["ACGTACGT", "AC-TNCGA", None, "acgt..gt"])
    g = pd.Series(["ACGTACGA", "ACGTACGT", "ACGT", "ACGTACGA"])
    al, mm = bcr.count_mismatches(q, g)
    np.testing.assert_array_equal(al, [8, 6, 0, 6])
    np.testing.assert_array_equal(mm, [1, 1, 0, 1])
    al, mm = bcr.count_mismatches(q, g, skip_ambiguous=False)  # the archive's rule: N against a base is a mismatch
    np.testing.assert_array_equal(al, [8, 7, 0, 6])
    np.testing.assert_array_equal(mm, [1, 2, 0, 1])
    al, mm = bcr.count_mismatches(q, g, start=2, end=6)
    np.testing.assert_array_equal(al, [5, 3, 0, 3])
    np.testing.assert_array_equal(mm, [0, 0, 0, 0])
    al, mm = bcr.count_mismatches(q, g, trim_5prime_nt=[2, 2, 0, 0])
    np.testing.assert_array_equal(al, [6, 4, 0, 6])
    with pytest.raises(ValueError):
        bcr.count_mismatches(q, g[:2])


def test_count_mismatches_is_row_order_invariant():
    rng = np.random.default_rng(3)
    g = ["".join(rng.choice(list("ACGT."), 40)) for _ in range(30)]
    q = ["".join(c if rng.random() > 0.1 else "T" for c in s) for s in g]
    al, mm = bcr.count_mismatches(q, g)
    perm = rng.permutation(30)
    al2, mm2 = bcr.count_mismatches([q[i] for i in perm], [g[i] for i in perm])
    np.testing.assert_array_equal(al[perm], al2)
    np.testing.assert_array_equal(mm[perm], mm2)


def _records(n_base=12, rng=0):
    """Allele A: unswitched baseline of 2 mismatches; allele B: too few baseline records."""
    rng = np.random.default_rng(rng)
    germ = "".join(rng.choice(list("ACGT"), 300))

    def mutate(k):
        s = list(germ)
        for p in rng.choice(300, k, replace=False):
            s[p] = {"A": "C", "C": "G", "G": "T", "T": "A"}[s[p]]
        return "".join(s)

    rows = []
    for _ in range(n_base):
        rows.append(("IGHV1-1*01", "IGHM", mutate(2), 50))
    rows += [("IGHV1-1*01", "IGHG1", mutate(6), 50), ("IGHV1-1*01", "IGHG1", mutate(4), 50)]
    rows += [("IGHV1-1*01", "IGHG1", mutate(9), 10)]  # too few reads
    rows += [("IGHV2-2*01", "IGHM", mutate(0), 50)] * 3 + [("IGHV2-2*01", "IGHA", mutate(8), 50)]
    d = pd.DataFrame(rows, columns=["v_call", "c_call", "v_sequence_alignment", "consensus_count"])
    d["v_germline_alignment"] = germ
    d["locus"] = "IGH"
    return bcr.isotype(d)


def test_shm_baseline_excess_evaluability():
    d, tab = bcr.shm(_records(), return_baseline=True)
    pd.testing.assert_frame_equal(d, bcr.shm(_records()))
    assert d.attrs["spaclone"]["shm"]["n_alleles_with_baseline"] == 1
    assert tab.loc["IGHV1-1*01", "baseline"] == 2
    assert np.isnan(tab.loc["IGHV2-2*01", "baseline"])  # 3 baseline records < 10
    a = d[d.v_call == "IGHV1-1*01"]
    assert (a.shm_aligned_nt == 300).all()
    np.testing.assert_array_equal(a.shm_excess.to_numpy()[-3:], [4, 2, 7])
    np.testing.assert_array_equal(a.shm_evaluable.to_numpy()[-3:], [True, True, False])
    np.testing.assert_array_equal(a.shm_mutated.to_numpy()[-3:], [True, False, False])
    assert not d[d.v_call == "IGHV2-2*01"].shm_evaluable.any()
    assert d.shm_per100.iloc[0] == pytest.approx(100 * 2 / 300)
    nb = bcr.shm(_records(), baseline=None, min_excess=5)
    assert nb[nb.v_call == "IGHV2-2*01"].shm_mutated.tolist() == [False, False, False, True]
    short = bcr.shm(_records(), min_aligned_nt=301)
    assert not short.shm_evaluable.any()


def test_shm_regions_trim_and_errors():
    d = _records()
    w = bcr.shm(d, region=(1, 150), baseline=None, min_aligned_nt=0)
    assert (w.shm_aligned_nt == 150).all()
    t = bcr.shm(d, trim_5prime_nt=100, baseline=None, min_aligned_nt=0)
    assert (t.shm_aligned_nt == 200).all()
    d2 = d.assign(j_sequence_alignment="ACGT", j_germline_alignment="ACGA")
    vj = bcr.shm(d2, region="V_J", baseline=None, min_aligned_nt=0)
    assert (vj.shm_aligned_nt == 304).all()
    np.testing.assert_array_equal(vj.shm_mismatches - bcr.shm(d2, baseline=None).shm_mismatches, 1)
    with pytest.raises(ValueError):
        bcr.shm(d, region="CDR3")
    with pytest.raises(KeyError):
        bcr.shm(d, region="V_J")
    with pytest.raises(KeyError, match="reads"):
        bcr.shm(d.drop(columns="consensus_count"))
    ok = bcr.shm(d.drop(columns="consensus_count"), min_reads=0)
    assert ok.shm_evaluable.sum() > 0


def test_tso_trim():
    air = pd.DataFrame({"v_sequence_alignment": ["TTCTGAGAGGCGGGAAACCCGGG", "ACGTACGT", None, "AC-GAGAGGGGTT"]})
    t = bcr.tso_trim(air)
    assert t.tolist() == [14 + 3, 0, 0, 12]  # match end + 3 non-templated bases (capped at the length); gaps removed


def test_call_shm_median_per_group():
    d = bcr.shm(_records())
    d["locus_id"] = [0] * 6 + [1] * 6 + [2, 2, 3, 4, 4, 4, 4]
    c = bcr.call_shm(d, by="locus")
    assert c.loc[2, "n_evaluable"] == 2 and c.loc[2, "excess"] == pytest.approx(3.0) and c.loc[2, "mutated"]
    assert c.loc[3, "n_evaluable"] == 0 and pd.isna(c.loc[3, "mutated"])
    assert not c.loc[0, "mutated"]
    assert bcr.call_shm(d, by="locus", min_excess=4).loc[2, "mutated"] == np.False_
    with pytest.raises(KeyError):
        bcr.call_shm(d.drop(columns="shm_excess"))


def _loci(n=2400, seed=0, planted=0.0):
    """Loci on a 1 mm square: IgG concentrated on the left third; mutation depends on isotype (IgG 0.35, IgM 0.05);
    ``planted`` adds isotype-independent mutation inside a disc."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1000, n)
    y = rng.uniform(0, 1000, n)
    p_g = np.where(x < 330, 0.6, 0.1)
    iso = np.where(rng.random(n) < p_g, "IgG", "IgM")
    p_m = np.where(iso == "IgG", 0.35, 0.05)
    disc = (x - 700) ** 2 + (y - 500) ** 2 < 90**2
    p_m = np.where(disc, np.minimum(1, p_m + planted), p_m)
    mutated = rng.random(n) < p_m
    return pd.DataFrame(
        {"x_um": x, "y_um": y, "isotype": iso, "mutated": mutated, "switched": iso == "IgG", "clone_id": np.arange(n)}
    )


@pytest.mark.skipif(not hasattr(tl, "hotspots"), reason="spaclone.tl.hotspots not installed")
def test_shm_hotspots_isotype_matched_null_and_power():
    loci = _loci()
    loci["mutated"] = loci.mutated.astype("boolean")
    loci.loc[::50, "mutated"] = pd.NA  # not evaluable: left out
    raw = bcr.shm_hotspots(loci, match_by=None, n_perm=49, rng=1, min_neff=5)
    iso = bcr.shm_hotspots(loci, n_perm=49, rng=1, min_neff=5)
    assert iso.summary["n_units"] == int(loci.mutated.notna().sum())
    assert iso.params["within"] == "isotype" and iso.params["match_by"] == "isotype"
    # the IgG-rich third looks "mutated" without the isotype expectation, but not with it
    assert raw.summary["n_hotspots"] >= 1 and (raw.table.x_um < 330).all()
    assert not (iso.table.x_um < 330).any()
    assert iso.summary["max_z"] < raw.summary["max_z"] - 1
    hot = bcr.shm_hotspots(_loci(planted=0.6, seed=2), n_perm=49, rng=1, min_neff=5)
    assert hot.summary["n_hotspots"] >= 1
    top = hot.table.iloc[0]
    assert abs(top.x_um - 700) < 120 and abs(top.y_um - 500) < 120
    again = bcr.shm_hotspots(_loci(planted=0.6, seed=2), n_perm=49, rng=1, min_neff=5)
    np.testing.assert_array_equal(again.null, hot.null)
    m = bcr.shm_map(loci, min_neff=5)
    assert m.n_units == iso.summary["n_units"]
    ax = pl.shm_share_map(hot, csr=hot, units=_loci(planted=0.6, seed=2))
    assert ax is not None


@pytest.mark.skipif(
    not (IGBLAST / "ncbi-igblast-1.22.0" / "bin" / "igblastn").exists(),
    reason="set SPACLONE_TEST_IGBLAST to a folder with ncbi-igblast-1.22.0/, database/ and imgt_mouse/",
)
def test_run_igblast_calls_the_germline_gene(tmp_path):
    fasta = (IGBLAST / "imgt_mouse" / "IGHV.fasta").read_text().split(">")[1]
    head, *body = fasta.splitlines()
    seq = "".join(body).replace(".", "").upper()
    gene = head.split("|")[1]
    res = bcr.run_igblast({"q1": seq}, organism="mouse", igblast_dir=IGBLAST / "ncbi-igblast-1.22.0",
                          germline_db=IGBLAST / "database", out=tmp_path / "out.tsv")  # fmt: skip
    assert res.sequence_id.tolist() == ["q1"]
    assert gene.split("*")[0] in str(res.v_call.iloc[0])
    assert (tmp_path / "out.tsv").exists()


def test_run_igblast_missing_executable_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="igblastn"):
        bcr.run_igblast({"q": "ACGT"}, executable=tmp_path / "no_igblastn")


@pytest.mark.skipif(not hasattr(tl, "hotspots"), reason="spaclone.tl.hotspots not installed")
def test_shm_hotspots_calibration_under_the_isotype_null():
    """Isotype-dependent mutation with isotypes clustered in space: maps with any hotspot <= alpha + tolerance."""
    reps = 20
    hits = sum(
        bcr.shm_hotspots(_loci(n=1200, seed=100 + s), n_perm=39, rng=s, min_neff=5).summary["n_hotspots"] > 0
        for s in range(reps)
    )
    assert hits / reps <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)


def test_shm_baseline_fallback_is_explicit():
    d = _records()
    tab = bcr.allele_baseline(d, bcr.shm(d, baseline=None).shm_mismatches, min_reads=30, min_aligned_nt=250)
    assert tab.loc["IGHV1-1*01", "source"] == "allele" and tab.loc["IGHV2-2*01", "source"] == "none"
    assert tab.loc["IGHV2-2*01", "n"] == 3 and np.isnan(tab.loc["IGHV2-2*01", "baseline"])
    zero = bcr.shm(d, baseline_fallback="zero")
    b = zero[zero.v_call == "IGHV2-2*01"]
    assert (b.shm_baseline_source == "zero").all() and (b.shm_baseline == 0).all() and b.shm_evaluable.all()
    assert b.shm_mutated.tolist() == [False, False, False, True]
    pooled = bcr.shm(d, baseline_fallback="pooled")
    assert (pooled[pooled.v_call == "IGHV2-2*01"].shm_baseline == 2).all()  # median of the 12 + 3 IgM records
    assert bcr.shm(d, baseline_fallback="pooled", return_baseline=True)[1].loc["IGHV2-2*01", "n_source"] == 15
    assert bcr.shm(d, baseline=None, return_baseline=True)[1] is None
    # the gene level: another allele of IGHV1-1 borrows the gene's median
    d3 = pd.concat([d, d.iloc[[12]].assign(v_call="IGHV1-1*02")], ignore_index=True)
    g = bcr.shm(d3, baseline_fallback="gene")
    row = g[g.v_call == "IGHV1-1*02"].iloc[0]
    assert row.shm_baseline_source == "gene" and row.shm_baseline == 2
    assert (bcr.shm(d).shm_baseline_source[d.v_call == "IGHV2-2*01"] == "none").all()
    assert (bcr.shm(d, baseline=None).shm_baseline_source == "off").all()
    with pytest.raises(ValueError, match="fallback"):
        bcr.allele_baseline(d, np.zeros(len(d)), fallback="median")


def test_attrs_hold_scalars_so_tables_concatenate_and_merge():
    """pandas >= 3 compares ``attrs`` in concat / merge: a DataFrame there raised "truth value ... ambiguous"."""
    d = bcr.shm(_records())
    for out in (d, bcr.call_isotype(d, by="v_call"), bcr.shm(_records(), region=(1, 150), baseline=None)):
        _plain(out.attrs)
        both = pd.concat([out, out])
        assert len(both) == 2 * len(out) and both.attrs == out.attrs
        assert len(out.merge(out, left_index=True, right_index=True, suffixes=("", "_r"))) == len(out)
    m = d.merge(d[["v_call"]].drop_duplicates(), on="v_call")
    assert len(m) == len(d)
    if int(pd.__version__.split(".")[0]) >= 3:  # pandas 2.x drops attrs on merge; pandas 3 keeps (and compares) them
        assert m.attrs == d.attrs


def test_shm_from_mixcr_alignments():
    from spaclone.io import parse_mixcr_alignment

    al = ["223|373|391|0|150|SA223GSC300T|1470.0", "0|366|387|34|399|DG16DA17SA199GI329T|3415.0", None,
          "10|260|300|0|250|SA20NSC50TSG70A|100.0"]  # fmt: skip
    air = pd.DataFrame({"v_call": "IGHV1-1*00", "c_call": "IGHG1", "locus": "IGH", "v_alignment_mixcr": al})
    r = bcr.shm(air, mixcr_alignment="v_alignment_mixcr", baseline=None, min_reads=0, min_aligned_nt=100)
    np.testing.assert_array_equal(r.shm_mismatches, [2, 1, 0, 2])
    np.testing.assert_array_equal(r.shm_aligned_nt, [150, 364, 0, 249])  # span - deletions - ambiguous
    np.testing.assert_array_equal(r.shm_evaluable, [True, True, False, True])
    np.testing.assert_array_equal(r.shm_mutated, [False, False, False, False])
    pv = parse_mixcr_alignment(air.v_alignment_mixcr)
    np.testing.assert_array_equal(pv.substitutions.fillna(0).to_numpy(dtype=int), r.shm_mismatches)
    lax = bcr.shm(air, mixcr_alignment="v_alignment_mixcr", baseline=None, min_reads=0, skip_ambiguous=False)
    assert lax.shm_mismatches.iloc[3] == 3 and lax.shm_aligned_nt.iloc[3] == 250
    trimmed = bcr.shm(air, mixcr_alignment="v_alignment_mixcr", baseline=None, min_reads=0, trim_5prime_nt=60)
    assert trimmed.shm_mismatches.iloc[3] == 1 and trimmed.shm_aligned_nt.iloc[3] == 190  # S20N, C50T trimmed
    j = air.assign(j_alignment_mixcr="0|40|50|300|340|SA5G|200.0")
    vj = bcr.shm(j, mixcr_alignment="v_alignment_mixcr", mixcr_j_alignment="j_alignment_mixcr", region="V_J",
                 baseline=None, min_reads=0)  # fmt: skip
    np.testing.assert_array_equal(vj.shm_mismatches - r.shm_mismatches, 1)
    with pytest.raises(ValueError, match="mixcr_j_alignment"):
        bcr.shm(j, mixcr_alignment="v_alignment_mixcr", region="V_J", baseline=None, min_reads=0)
    with pytest.raises(ValueError, match="AIRR"):
        bcr.shm(j, mixcr_alignment="v_alignment_mixcr", region=(1, 100), baseline=None, min_reads=0)


@pytest.mark.parametrize("reps", [30, pytest.param(200, marks=pytest.mark.slow)])
def test_shm_hotspots_clone_level_default_is_calibrated(reps):
    """The reviewer's case: mutated is a clone-level mark (0.6 IgG1 / 0.2 IgM clones), clones of 1-8 loci at random
    centres. Loci shuffled one by one within isotype rejected on 99% of maps; the default clone-level design stays at
    alpha (records of a clone 4 um apart are merged into one locus first)."""
    rng = np.random.default_rng(41)
    hits = hits_legacy = 0
    for _ in range(reps):
        n_cl = 80
        cx, cy = rng.uniform(0, 300, n_cl), rng.uniform(0, 300, n_cl)
        size = rng.integers(1, 9, n_cl)
        clone = np.repeat(np.arange(n_cl), size)
        iso = rng.choice(["IgM", "IgG1"], n_cl)
        mut = rng.random(n_cl) < np.where(iso == "IgG1", 0.6, 0.2)
        t = pd.DataFrame({"x_um": cx[clone] + rng.normal(0, 4, len(clone)),
                          "y_um": cy[clone] + rng.normal(0, 4, len(clone)), "clone_id": clone.astype(str),
                          "isotype": iso[clone], "mutated": mut[clone]})  # fmt: skip
        kw = {"n_perm": 39, "sigma_um": 25.0, "min_neff": 3.0, "rng": int(rng.integers(2**31))}
        hits += bcr.shm_hotspots(t, **kw).summary["n_hotspots"] > 0
        hits_legacy += bcr.shm_hotspots(t, blocks=None, **kw).summary["n_hotspots"] > 0
    assert hits / reps <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)
    assert hits_legacy / reps > 0.5
