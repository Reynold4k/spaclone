"""Bridges to established single-cell immune-repertoire and transcriptome packages.

Each submodule imports its package lazily, so ``import spaclone`` never requires them:

* :mod:`spaclone.ext.scicsr` — sciCSR (Ng et al., Nat Methods 2023): CSR potential, SHM score, transition models and
  transition-path-theory fluxes, on spatial units, plus a bridge to the original R package.
* :mod:`spaclone.ext.scirpy` — scirpy (scverse) on spatial loci or spots.
* :mod:`spaclone.ext.dandelion` — Dandelion clone definitions.
* :mod:`spaclone.ext.immcantation` — IgBLAST / Change-O / dowser round trips.
* :mod:`spaclone.ext.squidpy` — squidpy spatial statistics on receptor-derived labels.
* :mod:`spaclone.ext.threadfin` — Threadfin clone-state profiles on spatial loci, mapped back to tissue.
"""

from . import dandelion, immcantation, scicsr, scirpy, squidpy, threadfin

__all__ = ["dandelion", "immcantation", "scicsr", "scirpy", "squidpy", "threadfin"]
