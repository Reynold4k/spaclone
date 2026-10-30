import numpy as np
import pandas as pd
import pytest

from spaclone import bcr

_OTHER = {"A": "CGT", "C": "AGT", "G": "ACT", "T": "ACG"}


def _rec(germ, changes):
    s = list(germ)
    for pos, base in changes.items():
        s[pos] = base
    return "".join(s)


def test_mutations_known_answers_motifs_and_codon_positions():
    #        0123456789012345
    germ = "GGTACTGGAGCCCAAT"  # C at 4 in T-A-C-T (WRCY); C at 12 in C-C-C (SYC: S=C, Y=C); A at 13 in C-A (not WA)
    seq = _rec(germ, {4: "T", 12: "A", 13: "G", 15: "C"})
    air = pd.DataFrame({"sequence_alignment": [seq], "germline_alignment": [germ]})
    m = bcr.mutations(air).set_index("column")
    assert list(m.index) == [5, 13, 14, 16]
    assert m.loc[5, "WRCY/RGYW"] and not m.loc[5, "SYC/GRS"] and m.loc[5, "transition"]
    assert m.loc[5, "codon_position"] == 2
    assert m.loc[13, "SYC/GRS"] and not m.loc[13, "WRCY/RGYW"] and not m.loc[13, "transition"]
    assert not m.loc[14, "WA/TW"]  # A preceded by C
    assert not m.loc[16, "WA/TW"]  # T at the end: right context unknown
    assert m.loc[16, "substitution"] == "T>C"
    shifted = bcr.mutations(air, frame_offset=1).set_index("column")
    assert shifted.loc[5, "codon_position"] == 3


def test_motif_context_skips_imgt_gaps_and_unknown_bases():
    germ = "GGTA...CTGG"  # ungapped context of the C: T-A-C-T
    seq = "GGTA...TTGG"
    m = bcr.mutations(pd.DataFrame({"sequence_alignment": [seq], "germline_alignment": [germ]}))
    assert len(m) == 1 and m.column.iloc[0] == 8 and m["WRCY/RGYW"].iloc[0]
    assert m.codon_position.iloc[0] == 2  # IMGT gap columns count for the codon phase
    germ_n = "GGNACTGG"
    m2 = bcr.mutations(pd.DataFrame({"sequence_alignment": ["GGNATTGG"], "germline_alignment": [germ_n]}))
    assert not m2["WRCY/RGYW"].iloc[0]  # context with N: not classifiable


def _simulate(rng, n=60, k=6, hot_bias=0.0, length=300):
    """Random germlines; k mutations per record at positions drawn with weight 1 + hot_bias at WRCY / RGYW sites."""
    g = rng.integers(0, 4, (n, length))  # 0..3 = A C G T
    W, R, Y = np.isin(g, (0, 3)), np.isin(g, (0, 2)), np.isin(g, (1, 3))
    hot = np.zeros((n, length), dtype=bool)
    hot[:, 2:-1] |= (g[:, 2:-1] == 1) & W[:, :-3] & R[:, 1:-2] & Y[:, 3:]  # C of W-R-C-Y
    hot[:, 1:-2] |= (g[:, 1:-2] == 2) & R[:, :-3] & Y[:, 2:-1] & W[:, 3:]  # G of R-G-Y-W
    w = np.where(hot, 1.0 + hot_bias, 1.0)
    keys = rng.random((n, length)) ** (1.0 / w)  # weighted sampling without replacement (Efraimidis-Spirakis)
    pos = np.argsort(-keys, axis=1)[:, :k]
    q = g.copy()
    rows = np.repeat(np.arange(n), k)
    q[rows, pos.ravel()] = (g[rows, pos.ravel()] + rng.integers(1, 4, n * k)) % 4
    letters = np.array(list("ACGT"))
    return pd.DataFrame(
        {
            "sequence_alignment": ["".join(r) for r in letters[q]],
            "germline_alignment": ["".join(r) for r in letters[g]],
        }
    )


def test_spectrum_calibration_under_uniform_targeting():
    rng = np.random.default_rng(0)
    reps = 100
    hits = 0
    for _ in range(reps):
        r = bcr.spectrum(_simulate(rng), region=None, n_perm=99, rng=rng)
        hits += r.table.set_index("motif").loc["WRCY/RGYW", "p_value"] <= 0.05
    assert hits / reps <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)


def test_spectrum_power_and_tables():
    rng = np.random.default_rng(1)
    d = _simulate(rng, hot_bias=6.0)
    d["group"] = np.where(np.arange(len(d)) % 2 == 0, "a", "b")
    r = bcr.spectrum(d, region=None, n_perm=199, rng=2, alternative="greater")
    row = r.table.set_index("motif").loc["WRCY/RGYW"]
    assert row.ratio > 1.5 and row.p_value <= 0.01
    assert row.expected == pytest.approx(row.null_mean, rel=0.1)
    assert r.summary["mutations"] == 6 * len(d)
    assert r.substitutions["count"].sum() == 6 * len(d)
    assert len(r.substitutions) == 12 and r.codon_positions.mutations.sum() == 6 * len(d)
    assert 0.2 < r.summary["transition_share"] < 0.5  # uniform substitutions: 1/3 transitions
    g = bcr.spectrum(d, region=None, by="group", n_perm=49, rng=2)
    assert set(g.table.group) == {"a", "b"} and len(g.table) == 6
    a = bcr.spectrum(d, region=None, n_perm=49, rng=3, keep_null=True)
    b = bcr.spectrum(d, region=None, n_perm=49, rng=3, keep_null=True)
    np.testing.assert_array_equal(a.null, b.null)
    w = bcr.spectrum(d, region=(1, 150), n_perm=19, rng=0)
    assert w.summary["mutations"] < r.summary["mutations"]
    with pytest.raises(ValueError):
        bcr.spectrum(d, region="CDR9")
