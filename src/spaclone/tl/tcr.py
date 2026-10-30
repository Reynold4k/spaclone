"""Invariant T-cell receptor chains: iNKT and MAIT alpha chains.

Two innate-like T-cell populations carry an (almost) invariant TCR alpha chain, so a single receptor record identifies
the cell type, independent of the transcriptome:

* iNKT cells: mouse Valpha14-Jalpha18 (TRAV11-TRAJ18; junction CVVGDRGSALGRLHF), human Valpha24-Jalpha18
  (TRAV10-TRAJ18; junction CVVSDRGSTLGRLYF);
* MAIT cells: mouse Valpha19-Jalpha33 (TRAV1-TRAJ33; junction CAVRDSNYQLIW), human Valpha7.2 (TRAV1-2) joined to
  TRAJ33, TRAJ12 or TRAJ20 with a 12-residue CDR3 alpha.

The ``"canonical"`` rule requires the genes and the canonical junction (mouse; human iNKT) or the junction length
(human MAIT); the ``"genes"`` rule requires the V and J genes only. Mouse TRAV11 has duplicated copies named TRAV11D
and TRAV11N in some references; all three count. The beta chains of iNKT cells are biased (mouse TRBV13-2, TRBV29,
TRBV1 = Vbeta8.2, 7, 2; human TRBV25-1) but not specific; they are flagged separately for identity checks.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from spaclone import _schema as S
from spaclone.pp.clones import gene_name

__all__ = ["INKT_BETA_V", "INVARIANT_TCR", "invariant_tcr"]

#: Rules per species and cell type: chain, V genes, J genes, canonical junctions (or junction lengths).
INVARIANT_TCR: dict[str, dict[str, dict[str, Any]]] = {
    "mouse": {
        "iNKT": {
            "locus": "TRA",
            "v": ("TRAV11", "TRAV11D", "TRAV11N"),
            "j": ("TRAJ18",),
            "junction": ("CVVGDRGSALGRLHF",),
        },
        "MAIT": {"locus": "TRA", "v": ("TRAV1",), "j": ("TRAJ33",), "junction": ("CAVRDSNYQLIW",)},
    },
    "human": {
        "iNKT": {"locus": "TRA", "v": ("TRAV10",), "j": ("TRAJ18",), "junction": ("CVVSDRGSTLGRLYF",)},
        "MAIT": {"locus": "TRA", "v": ("TRAV1-2",), "j": ("TRAJ33", "TRAJ12", "TRAJ20"), "junction_length": (12,)},
    },
}
#: TRB V genes preferred (not used exclusively) by iNKT cells.
INKT_BETA_V: dict[str, tuple[str, ...]] = {"mouse": ("TRBV13-2", "TRBV29", "TRBV1"), "human": ("TRBV25-1",)}


def invariant_tcr(
    air: pd.DataFrame,
    *,
    species: Literal["mouse", "human"] = "mouse",
    rule: Literal["canonical", "genes"] = "canonical",
    rules: dict[str, dict[str, Any]] | None = None,
    key_added: str = "invariant",
    locus_key: str = S.LOCUS,
    v_key: str = S.V_CALL,
    j_key: str = S.J_CALL,
    junction_key: str = S.JUNCTION_AA,
) -> pd.DataFrame:
    """Flag iNKT and MAIT alpha chains (and iNKT-type beta V genes).

    Parameters
    ----------
    air
        Receptor table with V calls, J calls and junctions (amino acids). Calls may carry alleles or several
        comma-separated genes; the first gene is used (:func:`spaclone.pp.gene_name`).
    species
        ``"mouse"`` or ``"human"`` (rules in :data:`INVARIANT_TCR`).
    rule
        ``"canonical"``: V gene, J gene and the canonical junction (human MAIT: junction length 12); ``"genes"``: V and
        J gene only (catches junction variants and sequencing errors, and some non-invariant cells).
    rules
        Custom rules replacing :data:`INVARIANT_TCR` for ``species``: ``{name: {"locus", "v", "j", "junction" or
        "junction_length"}}``.
    key_added
        Column with the cell-type name (``"iNKT"``, ``"MAIT"`` or missing).
    locus_key, v_key, j_key, junction_key
        Column names. Without a chain column the chain is read from the first three letters of the V gene.

    Returns
    -------
    pandas.DataFrame
        A copy of ``air`` with ``key_added`` and one boolean column per rule named in lower case (``inkt``, ``mait``),
        plus ``inkt_beta_v``: TRB records whose V gene is one the iNKT beta chains prefer (:data:`INKT_BETA_V`;
        conventional T cells use these genes too, so this is an identity check, not a call).

    References
    ----------
    Lantz & Bendelac 1994, doi:10.1084/jem.180.3.1097 (mouse Valpha14-Jalpha18); Dellabona et al. 1994,
    doi:10.1084/jem.180.3.1171 (human Valpha24-JalphaQ); Tilloy et al. 1999, doi:10.1084/jem.189.12.1907 (MAIT
    Valpha19 / Valpha7.2-Jalpha33); Reantragoon et al. 2013, doi:10.1084/jem.20130958 (human MAIT TRAJ33 / 12 / 20);
    Bendelac et al. 2007, doi:10.1146/annurev.immunol.25.022106.141711 (iNKT beta-chain bias).
    """
    if species not in INVARIANT_TCR and rules is None:
        raise ValueError(f"species must be one of {sorted(INVARIANT_TCR)}, got {species!r}")
    if rule not in ("canonical", "genes"):
        raise ValueError(f"rule must be 'canonical' or 'genes', got {rule!r}")
    missing = [c for c in (v_key, j_key, junction_key) if c not in air.columns]
    if missing:
        raise KeyError(f"invariant_tcr: column(s) {missing} not in the table")
    table = INVARIANT_TCR.get(species, {}) if rules is None else rules
    out = air.copy()
    v = gene_name(air[v_key])
    j = gene_name(air[j_key])
    jx = air[junction_key].astype("string")
    if locus_key in air.columns:
        locus = air[locus_key].astype("string").str.upper()
    else:
        locus = v.str[:3].str.upper()
    label = pd.Series(pd.NA, index=air.index, dtype="string")
    for name, r in table.items():
        hit = (locus == r["locus"]) & v.isin(list(r["v"])) & j.isin(list(r["j"]))
        if rule == "canonical":
            if "junction" in r:
                hit &= jx.isin(list(r["junction"]))
            if "junction_length" in r:
                hit &= jx.str.len().isin(list(r["junction_length"]))
        hit = hit.fillna(False).to_numpy(dtype=bool)
        out[name.lower()] = hit
        label = label.mask(pd.Series(hit, index=air.index) & label.isna(), name)
    out[key_added] = label
    beta = INKT_BETA_V.get(species, ())
    out["inkt_beta_v"] = ((locus == "TRB") & v.isin(list(beta))).fillna(False).to_numpy(dtype=bool)
    return out
