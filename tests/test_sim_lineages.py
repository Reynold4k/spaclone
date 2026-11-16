"""Tests of the simulated receptor sequences (sim.synthetic_chip(sequences=True)) and of sim.score_lineages."""

import numpy as np
import pandas as pd
import pytest

from spaclone import bcr, datasets, sim
from spaclone.bcr.lineage import read_trees
from spaclone.sim.lineages import _germline_set

SMALL = {
    "size_um": 300.0,
    "tissue_radius_um": 120.0,
    "spot_um": 4.0,
    "t_zone_radius_um": 35.0,
    "follicle_width_um": 50.0,
    "n_gc": 2,
    "gc_radius_um": 22.0,
    "n_pc_foci": 2,
    "pc_radius_um": 18.0,
    "sink_size": 40,
    "cell_density": 0.03,
}


@pytest.fixture(scope="module")
def clean():
    """A small chip with planted lineages and no sequencing errors."""
    return sim.synthetic_chip(rng=0, sequences=True, seq_error_rate=0.0, **SMALL)


@pytest.fixture(scope="module")
def clean_result(clean):
    return bcr.lineage_workflow(clean, rng=0, min_tips=5)


def test_sequences_leave_the_rest_of_the_chip_unchanged():
    plain = sim.synthetic_chip(rng=3, **SMALL)
    seq = sim.synthetic_chip(rng=3, sequences=True, **SMALL)
    for c in plain.air.columns:
        pd.testing.assert_series_equal(seq.air[c], plain.air[c])
    pd.testing.assert_frame_equal(seq.gex.obs, plain.gex.obs)
    assert (seq.gex.X != plain.gex.X).nnz == 0
    added = [c for c in seq.air.columns if c not in plain.air.columns]
    assert added == ["sequence_id", "sequence", "v_sequence_start", "v_sequence_end", "v_germline_start",
                     "v_germline_end", "v_sequence_alignment", "v_germline_alignment", "j_sequence_start",
                     "j_sequence_end", "true_genotype", "true_mutations", "seq_errors", "seq_error_sites"]  # fmt: skip
    again = sim.synthetic_chip(rng=3, sequences=True, **SMALL)
    pd.testing.assert_frame_equal(again.air, seq.air)
    assert again.meta["lineage_trees"] == seq.meta["lineage_trees"]
    other = sim.synthetic_chip(rng=4, sequences=True, **SMALL)
    assert not other.air["sequence"].equals(seq.air["sequence"])
    with pytest.raises(ValueError, match="realistic chip"):
        sim.synthetic_chip(rng=0, sequences=True, legacy=True)
    with pytest.raises(ValueError, match="seq_error_rate"):
        sim.synthetic_chip(rng=0, sequences=True, seq_error_rate=2.0)
    ex = datasets.synthetic_example(sequences=True, **SMALL)
    pd.testing.assert_frame_equal(ex.air, sim.synthetic_chip(rng=0, sequences=True, **SMALL).air)


def test_contigs_carry_their_alignments_in_airr_coordinates(clean):
    air = clean.air
    germ = _germline_set()
    for r in air.sample(200, random_state=1).itertuples(index=False):
        seq = r.sequence
        v0, v1 = r.v_sequence_start - 1, r.v_sequence_end  # AIRR: 1-based, closed
        assert seq[v0:v1] == r.v_sequence_alignment  # no indels are simulated
        vg = germ["v"][r.v_call.split("*")[0]]
        assert vg[r.v_germline_start - 1 : r.v_germline_end] == r.v_germline_alignment
        js = seq.find(r.junction)
        assert js == v1 - 3  # the V alignment ends with the conserved cysteine codon, the first junction codon
        assert r.j_sequence_start - 1 == js + len(r.junction) - 3  # the J alignment starts at the last junction codon
        assert seq[r.j_sequence_start - 1 : r.j_sequence_end] == r.junction[-3:] + germ["fr4"][r.j_call.split("*")[0]]
        if r.v_germline_start == 1:
            assert seq[:v0] == germ["leader"][r.v_call.split("*")[0]]


def test_planted_mutations_follow_the_cell_types(clean):
    air, meta = clean.air, clean.meta
    full = air[air.v_germline_start == 1]
    diff = [sum(a != b for a, b in zip(s, g, strict=True)) for s, g in
            zip(full.v_sequence_alignment, full.v_germline_alignment, strict=True)]  # fmt: skip
    assert (np.asarray(diff) == full.true_mutations.to_numpy()).all()  # no errors in this chip
    naive = air[(air.true_cell_type == "B") & air.locus.isin(["IGH", "IGK", "IGL"])]
    assert len(naive) and (naive.true_mutations == 0).all()  # naive B cells keep their germline
    assert (air[air.locus.isin(["TRA", "TRB"])].true_mutations == 0).all()  # no hypermutation in T cells
    gc = air[(air.true_compartment == "germinal centre") & (air.locus == "IGH")]
    assert gc.true_mutations.mean() > 2
    heavy, light = (air[air.locus == loc].groupby("lineage").true_mutations.mean() for loc in ("IGH", "IGK"))
    mutated = meta["genotypes"][meta["genotypes"].mutations > 0].lineage.unique()
    common = sorted(set(heavy.index) & set(light.index) & set(mutated))
    assert heavy[common].mean() > light[common].mean()  # light chains mutate at half the rate
    # every record of one cell and chain carries that cell's genotype; one genotype = one sequence
    per_cell = air.groupby(["true_cell", "locus"]).true_genotype.nunique()
    assert (per_cell == 1).all()
    trees = read_trees(meta["lineage_trees"])
    assert set(trees) <= set(meta["lineages"].lineage)
    for lid, tree in trees.items():
        tips = {t for t in tree.tips() if t != "Germline"}
        assert tips <= set(meta["genotypes"].genotype) and all(t.startswith(lid) for t in tips)


def test_sequencing_errors_and_truncation_are_recorded():
    sa = sim.synthetic_chip(rng=1, sequences=True, seq_error_rate=0.01, truncated_fraction=0.3, **SMALL)
    air = sa.air
    assert air.seq_errors.mean() == pytest.approx(0.01 * 290, rel=0.15)
    r = air[air.seq_errors > 0].iloc[0]
    for site in r.seq_error_sites.split(";"):
        p, b = site.split(":")
        assert r.v_sequence_alignment[int(p) - (r.v_germline_start - 1)] == b
    trunc = air.v_germline_start > 1
    assert abs(trunc.mean() - 0.3) < 0.05
    assert air.loc[trunc, "v_germline_start"].between(31, 150).all()
    sa0 = sim.synthetic_chip(rng=1, sequences=True, seq_error_rate=0.0, **SMALL)
    assert (sa0.air.seq_errors == 0).all() and (sa0.air.seq_error_sites == "").all()


def test_lineage_workflow_recovers_the_planted_trees(clean, clean_result):
    """With no sequencing errors every tip carries exactly its planted mutations and every tree has the planted
    topology (Robinson-Foulds 0) on the genotypes it shows; ranks follow the most mutated tip."""
    res = clean_result
    assert res.params["noise_min_reads"] == 1 and res.params["min_reads"] == 1  # molecule-level records
    sc = sim.score_lineages(res, clean)
    assert len(sc) == len(res.trees["exact"]) >= 3
    assert (sc.purity == 1).all() and (sc.rf == 0).all()
    assert sc.depth_cor.min() > 0.95
    s = sc.attrs["summary"]
    assert s["tips_exact"] >= 0.98 * s["tips"] and s["false_mutations"] == 0
    assert s["recovered_mutations"] >= 0.99 * s["planted_mutations"]
    assert s["records_exact_mutations"] > 0.98
    sel = res.selection[res.selection.grouping == "exact"].sort_values("rank")
    assert sel.max_mutations.is_monotonic_decreasing and res.params["order"] == "max_mutation"
    gen = res.records.set_index("sequence_id").clone_exact
    lin = clean.air.set_index("sequence_id").lineage.reindex(gen.index)
    assert (gen.groupby(lin).nunique() == 1).all()  # one exact clone per planted lineage (junctions never mutate)


def test_errors_seen_once_are_reverted_and_scores_see_them():
    sa = sim.synthetic_chip(rng=2, sequences=True, seq_error_rate=0.001, **SMALL)
    res = bcr.lineage_workflow(sa, rng=0, min_tips=5, groupings=("exact",))
    s = sim.score_lineages(res, sa).attrs["summary"]
    assert s["errors_in_full_length_records"] > 500
    assert s["errors_reverted"] >= 0.9 * s["errors_in_full_length_records"]  # the rest: the same error in two spots
    assert s["false_mutations"] > 0  # ... which the scores report as mutations the cells do not carry
    # with the archive's 30-read floor of contigs nothing survives on molecule-level records: every mutation is lost
    strict = bcr.lineage_workflow(sa, rng=0, min_tips=5, groupings=("exact",), noise_min_reads=30)
    assert strict.records.mutations_exact.max() == 0


def test_score_lineages_needs_a_simulated_section(clean_result):
    sa = sim.synthetic_chip(rng=0, **SMALL)
    with pytest.raises(ValueError, match="sequences=True"):
        sim.score_lineages(clean_result, sa)
    with pytest.raises(ValueError, match="grouping"):
        sim.score_lineages(clean_result, sa, grouping="clonotype")
