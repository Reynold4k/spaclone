"""Tests of spaclone.bcr.csr_dynamics: CSR potential, SHM score, IGHC definitions and BAM transcript counting."""

import struct
import zlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spaclone import bcr
from spaclone.bcr import csr_dynamics as cd


def _air():
    return pd.DataFrame(
        {
            "spot_id": ["a", "a", "b", "c", "c", "d", "e"],
            "x_um": [0, 0, 10, 20, 20, 30, 40.0],
            "y_um": [0.0] * 7,
            "locus": ["IGH", "IGH", "IGH", "IGH", "IGK", "IGK", "IGH"],
            "clone_id": list("abcdefg"),
            "c_call": ["IGHM*01", "IGHG1", "IGHA", None, "IGKC", "IGKC", "IGHD"],
            "productive": [True] * 7,
        }
    )


# ================================================================================================ isotype positions
def test_positions_order_aliases_and_class_only():
    order, pos = cd._species_order("mouse", None)
    calls = pd.Series(["IGHM", "IGHG3", "IGHG1*01", "IGHG2B", "IGHG2A", "IGHG2C", "IGHE", "IGHA", "IGHD", "IGHG", None])
    p = cd._positions(calls, species="mouse", pos=pos, order=order, class_only="lower")
    np.testing.assert_array_equal(p[:8], [0, 1, 2, 3, 4, 4, 5, 6])  # IgG2a shares the IgG2c position
    assert np.isnan(p[8])  # IgD: no CSR position
    assert p[9] == 1  # class-only IgG -> its 5'-most gene (IgG3)
    assert np.isnan(p[10])
    p2 = cd._positions(calls, species="mouse", pos=pos, order=order, class_only="missing")
    assert np.isnan(p2[9])
    ho, hp = cd._species_order("human", None)
    ph = cd._positions(
        pd.Series(["IGHA1", "IGHA2", "IgG4", "A1", "G2"]), species="human", pos=hp, order=ho, class_only="lower"
    )
    np.testing.assert_array_equal(ph, [3, 7, 5, 3, 4])  # sciCSR short labels are understood too


# ================================================================================================ CSR potential
def test_furthest_highest_and_spatial_imputation_hand_computed():
    p = bcr.csr_potential(_air(), impute="spatial", k=2)
    # a: IgM + IgG1 -> furthest IgG1 (2); b: IgA (6); c: no heavy isotype, neighbours a (2), b (6), d (-) -> tie,
    # the 5'-most wins (sciCSR's which.max(table)); d, e (IgD only): no isotyped neighbour -> IgM fallback
    assert p.isotype.tolist() == ["IgG1", "IgA", "IgG1", "IgM", "IgM"]
    np.testing.assert_array_equal(p.representative_p, [2, 6, 2, 0, 0])
    np.testing.assert_array_equal(p.imputed, [False, False, True, True, True])
    np.testing.assert_allclose(p.csr_pot, [2 / 6, 1, 2 / 6, 0, 0])
    h = bcr.csr_potential(_air(), impute=None, fallback=None, mode="highest")
    assert h.loc["a", "isotype"] == "IgG1"  # tie between IgM and IgG1 -> the most 3'
    assert h.loc[["c", "d", "e"], "representative_p"].isna().all()


def test_explicit_adjacency_matches_spatial_knn():
    air = _air()
    a = bcr.csr_potential(air, impute="spatial", k=2)
    adj = cd._knn_adjacency(a[["x_um", "y_um"]].to_numpy(), 2, None)
    b = bcr.csr_potential(air, impute=adj)
    pd.testing.assert_series_equal(a.isotype, b.isotype)


def test_empirical_potential_with_sterile_counts():
    st = pd.DataFrame({"IgM_S": [1, 0, 2], "IgG1_S": [0, 3, 0], "IgM_P": [5, 0, 1]}, index=["a", "b", "c"])
    p = bcr.csr_potential(_air(), impute="spatial", k=2, sterile=st, sterile_norm="counts")
    np.testing.assert_allclose(p.total_s, [1, 3, 2, 0, 0])
    raw = np.sqrt(np.array([2, 6, 2, 0, 0]) ** 2 + np.array([1, 3, 2, 0, 0]) ** 2)
    np.testing.assert_allclose(p.csr_pot_raw, raw)
    np.testing.assert_allclose(p.csr_pot, (raw - raw.min()) / (raw.max() - raw.min()))
    same = bcr.csr_potential(_air().assign(c_call="IGHM"), impute=None)
    assert np.all(same.csr_pot == 0)  # a constant raw score scales to 0, as sciCSR's NA -> 0


def test_seurat_scaling_formula():
    rng = np.random.default_rng(1)
    cnt = rng.poisson(2, (40, 6)).astype(float)
    cnt[3] = 0
    lib = rng.uniform(200, 2000, 40)
    z = cd._seurat_scale(cnt, lib)
    tot = cnt.sum(1, keepdims=True)
    x = np.where(tot > 0, np.log1p(cnt / np.where(tot > 0, tot, 1) * 1e4), 0)
    design = np.column_stack([np.ones(40), lib])
    resid = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ref = (resid - resid.mean(0)) / resid.std(0, ddof=1)
    ref = np.minimum(ref, 10.0)
    np.testing.assert_allclose(z, ref, atol=1e-10)
    spike = np.zeros((200, 3))
    spike[0, 0] = 100
    spike[:, 2] = 1
    zs = cd._seurat_scale(spike, None)
    assert zs[0, 0] == 10.0 and zs[1:, 0].min() > -10  # only the upper tail is clipped
    assert np.all(zs[:, 1] == 0)  # zero variance -> 0, as Seurat


def test_reference_based_potential_nnls():
    sig = pd.DataFrame(
        {"naive": [2.0, 1.0, 0.0, 0.0], "switched": [0.0, 0.0, 1.0, 2.0]},
        index=["Ighm-S", "Ighm-P", "Igha-S", "Igha-P"],
    )
    st = pd.DataFrame(
        {"IgM_S": [9, 0, 4], "IgM_P": [5, 0, 2], "IgA_S": [0, 6, 3], "IgA_P": [0, 9, 4]}, index=["a", "b", "c"]
    )
    p = bcr.csr_potential(
        _air().query("spot_id in ['a', 'b', 'c']"), impute=None, sterile=st, reference=sig, library_size=None
    )
    assert p.loc["a", "csr_pot"] == pytest.approx(0.0)  # most naive profile
    assert p.loc["b", "csr_pot"] == pytest.approx(1.0)  # least naive profile
    assert 0 < p.loc["c", "csr_pot"] < 1
    assert p.naive_weight.idxmax() == "a"
    with pytest.raises(ValueError, match="needs sterile"):
        bcr.csr_potential(_air(), reference=sig)


def test_transcript_isotypes_furthest_and_highest():
    st = pd.DataFrame({"IgM_P": [3, 0, 0], "IgG1_P": [1, 0, 0], "IgA_P": [0, 0, 2]}, index=["a", "b", "c"])
    units = pd.DataFrame({"x_um": [0.0, 10, 20], "y_um": [0.0, 0, 0]}, index=["a", "b", "c"])
    f = bcr.csr_potential(
        units, isotype_source="transcripts", sterile=st, sterile_norm="counts", impute="spatial", k=1, unit_key=None
    )
    assert f.isotype.tolist() == ["IgG1", "IgG1", "IgA"]  # b imputed: tie IgG1 / IgA -> 5'-most
    h = bcr.csr_potential(units, isotype_source="transcripts", sterile=st, mode="highest", impute=None, unit_key=None)
    assert h.loc["a", "isotype"] == "IgM"


def test_anndata_input_and_order_invariance():
    a = ad.AnnData(obs=pd.DataFrame({"iso": ["IgM", "IgG2b", None, "IgA"]}, index=list("wxyz")))
    a.obsm["spatial"] = np.array([[0, 0], [10, 0], [20, 0], [30, 0.0]])
    p = bcr.csr_potential(a, isotype_key="iso", impute="spatial", k=1)
    assert p.index.tolist() == list("wxyz")
    assert p.loc["y", "isotype"] == "IgG2b"  # neighbours x (IgG2b) and z (IgA): tie -> 5'-most
    air = _air()
    r = bcr.csr_potential(air.iloc[::-1], impute="spatial", k=2).loc[["a", "b", "c", "d", "e"]]
    pd.testing.assert_frame_equal(r, bcr.csr_potential(air, impute="spatial", k=2), check_like=True)
    t = bcr.csr_potential(air.assign(x_um=air.x_um + 500), impute="spatial", k=2)
    np.testing.assert_array_equal(t.csr_pot, bcr.csr_potential(air, impute="spatial", k=2).csr_pot)


# ================================================================================================ SHM score
def test_shm_score_identity_percent_and_aggregation():
    sh = pd.DataFrame(
        {
            "cell": ["a", "b", "c", "a"],
            "locus": ["IGH"] * 4,
            "v_identity": [100.0, 95.0, np.nan, 98.0],
            "consensus_count": [5, 3, 1, 10],
        }
    )
    m = bcr.shm_score(sh, unit_key="cell")
    np.testing.assert_allclose(m.shm_raw.to_numpy()[:2], [0.01, 0.05])
    np.testing.assert_allclose(m.shm, [0.0, 1.0, 0.0])  # c has no identity -> 0 (sciCSR)
    t = bcr.shm_score(sh, unit_key="cell", agg="top_reads", fill=None)
    assert t.loc["a", "shm_raw"] == pytest.approx(0.02) and np.isnan(t.loc["c", "shm"])
    ex = sh.assign(shm_excess=[0, 6, 1, 3], shm_evaluable=[True, True, True, False])
    e = bcr.shm_score(ex, unit_key="cell", method="excess")
    np.testing.assert_allclose(e.shm_raw, [0, 6, 1])


# ================================================================================================ definitions
def test_ighc_definitions_layout():
    d = bcr.ighc_definitions("mouse", assembly="GRCm39")
    assert d.name.tolist()[0] == "VDJ" and len(d) == 1 + 2 * 7
    c = d[d.kind == "C"].set_index("name")
    s = d[d.kind == "sterile"].set_index("name")
    assert (s.start == c.end + 1).all()  # sterile region just 5' of each gene (minus strand)
    assert s.loc["IgM", "end"] == d.loc[0, "start"] - 1  # cut at the J genes
    assert (s.end - s.start + 1 <= 10_000).all()
    m38 = bcr.ighc_definitions("mouse", assembly="GRCm38")
    assert (m38.start.to_numpy()[1:] - d.start.to_numpy()[1:] == 36380).all()
    with pytest.raises(ValueError, match="not built in"):
        bcr.ighc_definitions("mouse", assembly="GRCh38")


def test_classify_molecule_rules():
    genes = ["IgM", "IgG1", "IgG2b"]
    cls = {"IgM": "IgM", "IgG1": "IgG", "IgG2b": "IgG"}
    cols = [f"{g}_C" for g in genes] + [f"{g}_I" for g in genes] + ["VDJ"]
    mol = pd.DataFrame(
        [
            [2, 0, 0, 0, 0, 0, 1],  # C + VDJ -> productive IgM
            [2, 0, 0, 0, 0, 0, 0],  # C only -> C-only IgM
            [0, 0, 0, 1, 0, 0, 0],  # I only -> sterile IgM
            [0, 1, 0, 0, 1, 0, 0],  # C and I agree, no VDJ -> sterile IgG1
            [1, 1, 0, 0, 0, 0, 0],  # two classes in C -> unclassified
            [0, 1, 1, 0, 0, 0, 0],  # IgG1 / IgG2b tie in C -> unclassified
            [0, 2, 0, 0, 0, 1, 0],  # I says IgG2b, C says IgG1 -> unclassified
            [0, 0, 0, 0, 1, 1, 1],  # I tie, no C, VDJ -> unclassified
            [0, 0, 0, 0, 0, 0, 3],  # VDJ only -> unclassified
        ],
        columns=cols,
    )
    got = cd._classify(mol, genes, cls).tolist()
    assert got == ["IgM_P", "IgM_C", "IgM_S", "IgG1_S", None, None, None, None, None]


# ================================================================================================ BAM reading
def _bgzf(data: bytes) -> bytes:
    comp = zlib.compressobj(6, zlib.DEFLATED, -15)
    c = comp.compress(data) + comp.flush()
    head = struct.pack("<BBBBIBBHBBHH", 31, 139, 8, 4, 0, 0, 255, 6, 66, 67, 2, len(c) + 25)
    return head + c + struct.pack("<II", zlib.crc32(data) & 0xFFFFFFFF, len(data))


def _reg2bin(beg, end):
    """SAM specification's bin of a 0-based half-open interval."""
    end -= 1
    for shift, first in ((14, 4681), (17, 585), (20, 73), (23, 9), (26, 1)):
        if beg >> shift == end >> shift:
            return first + (beg >> shift)
    return 0


def _record(ref, pos0, name, cigar, tags, flag=0):
    cig = b"".join(struct.pack("<I", ln << 4 | op) for op, ln in cigar)
    tg = b"".join(k.encode() + b"Z" + v.encode() + b"\x00" for k, v in tags.items())
    ref_len = sum(ln for op, ln in cigar if op in (0, 2, 3, 7, 8))
    bin_ = _reg2bin(pos0, pos0 + ref_len)
    body = struct.pack("<iiBBHHHIiii", ref, pos0, len(name) + 1, 60, bin_, len(cigar), flag, 0, -1, -1, 0)
    body += name.encode() + b"\x00" + cig + tg
    return struct.pack("<i", len(body)) + body


def _write_bam(path, records):
    hdr = b"@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:12\tLN:200000\n"
    data = b"BAM\x01" + struct.pack("<i", len(hdr)) + hdr + struct.pack("<i", 1)
    data += struct.pack("<i", 3) + b"12\x00" + struct.pack("<i", 200000)
    data += b"".join(records)
    eof = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")
    path.write_bytes(_bgzf(data) + eof)


def _toy_defs():
    return pd.DataFrame(
        {
            "name": ["VDJ", "IgM", "IgM", "IgG1", "IgG1"],
            "kind": ["VDJ", "C", "sterile", "C", "sterile"],
            "chrom": ["12"] * 5,
            "start": [20001, 10001, 12001, 1001, 3001],
            "end": [30000, 12000, 15000, 3000, 6000],
        }
    )


def _toy_records():
    m = 0  # CIGAR op M (3 = N, a splice gap); records in coordinate order
    return [
        _record(0, 1500, "r1", [(m, 50)], {"CB": "AAA", "UB": "u1"}),  # IgG1 C
        # read spanning the IgG1 sterile region only through an N gap: not in that region
        _record(0, 2500, "r8", [(m, 20), (3, 5000), (m, 20)], {"CB": "GGG", "UB": "u5"}),
        _record(0, 3500, "r2", [(m, 50)], {"CB": "AAA", "UB": "u1"}),  # IgG1 I -> sterile IgG1
        _record(0, 10200, "r6", [(m, 40)], {"CB": "CCC", "UB": "u4"}),  # C only -> C-only IgM
        _record(0, 10300, "r7", [(m, 40)], {"CB": "CCC"}),  # no UMI -> skipped
        _record(0, 10500, "r3", [(m, 50)], {"CB": "AAA", "UB": "u2"}),  # IgM C
        # spliced read: blocks in IgM C and in VDJ (the gap is an N), counted in both
        _record(0, 11900, "r5", [(m, 30), (3, 8100), (m, 30)], {"CB": "CCC", "UB": "u3"}),
        _record(0, 25000, "r4", [(m, 50)], {"CB": "AAA", "UB": "u2"}),  # VDJ -> productive IgM
    ]


def test_ighc_transcripts_python_reader(tmp_path):
    bam = tmp_path / "toy.bam"
    _write_bam(bam, _toy_records())
    counts, mol = bcr.ighc_transcripts(bam, definitions=_toy_defs(), backend="python", return_molecules=True)
    assert counts.loc["AAA", "IgG1_S"] == 1 and counts.loc["AAA", "IgM_P"] == 1
    assert counts.loc["CCC", "IgM_P"] == 1 and counts.loc["CCC", "IgM_C"] == 1
    assert counts.loc["GGG", "IgG1_C"] == 1  # C blocks only; the N gap does not touch the sterile region
    assert counts.to_numpy().sum() == 5
    m = mol.set_index(["unit", "umi"])
    assert m.loc[("CCC", "u3"), ["IgM_C", "VDJ"]].tolist() == [1, 1]
    assert ("CCC", "None") not in m.index
    spots = pd.DataFrame(
        {"spot_id": ["s1", "s1", "s2"], "x_um": [1.0, 1.0, 5.0], "y_um": [2.0, 2.0, 6.0]}, index=["AAA", "CCC", "GGG"]
    )
    sp = bcr.ighc_transcripts(bam, definitions=_toy_defs(), backend="python", spots=spots)
    assert sp.loc["s1", "IgM_P"] == 2 and sp.loc["s2", "x_um"] == 5.0


def test_ighc_transcripts_pysam_parity(tmp_path):
    pysam = pytest.importorskip("pysam")
    bam = tmp_path / "toy.bam"
    _write_bam(bam, _toy_records())
    pysam.index(str(bam))
    a = bcr.ighc_transcripts(bam, definitions=_toy_defs(), backend="python")
    b = bcr.ighc_transcripts(bam, definitions=_toy_defs(), backend="pysam")
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_ighc_transcripts_definition_errors(tmp_path):
    bam = tmp_path / "toy.bam"
    _write_bam(bam, _toy_records())
    with pytest.raises(ValueError, match="'VDJ', 'C' and 'sterile'"):
        bcr.ighc_transcripts(bam, definitions=_toy_defs().query("kind != 'VDJ'"), backend="python")
    with pytest.raises(ValueError, match="x_tag and y_tag"):
        bcr.ighc_transcripts(bam, definitions=_toy_defs(), backend="python", x_tag="Cx")
    assert sparse.issparse(cd._knn_adjacency(np.zeros((3, 2)), 2, None))
