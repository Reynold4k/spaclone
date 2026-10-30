# Install

spaclone runs on Python 3.11, 3.12 and 3.13.

## One command

```bash
pip install "spaclone[full]"
```

This installs spaclone and every optional Python package it can work with: scirpy, squidpy, SpatialData, Dandelion,
CellRank, Change-O, Threadfin, OLGA, Leiden and the dataset downloader. For the core alone, leave out `[full]`:

```bash
pip install spaclone
```

The core runs all of spaclone's own analyses (readers, QC and filtering, the transcriptome, every location-trust test,
class switching, hypermutation, lineage trees, repertoire and concentration statistics, plots). The extras add the
bridges to other packages ([](ecosystem.md)).

Then ask spaclone what works:

```python
import spaclone

spaclone.check_install()   # every optional feature: yes / no / broken, and the command that enables it
spaclone.show_versions()   # versions of Python, spaclone, dependencies and external programs
```

A package that is installed but fails to import (for example sc-dandelion 0.5 with matplotlib 3.9 or newer) is
reported as `broken`, with its error.

## Pick extras

Install one or several extras in the same way, for example `pip install "spaclone[scirpy,squidpy]"`.

| Extra | Installs | Licence | Enables |
|---|---|---|---|
| `scirpy` | scirpy, awkward, mudata | BSD-3-Clause | {mod}`spaclone.ext.scirpy`; {func}`~spaclone.io.to_scirpy` / {func}`~spaclone.io.from_scirpy`; the scirpy backend of {func}`~spaclone.tl.clonotype_network` |
| `squidpy` | squidpy | BSD-3-Clause | {mod}`spaclone.ext.squidpy`; {func}`~spaclone.gex.spatially_variable_genes` |
| `spatialdata` | spatialdata, shapely | BSD-3-Clause | {func}`~spaclone.io.to_spatialdata`; fast polygon lookup in {func}`~spaclone.pp.assign_to_cells` |
| `scverse` | the three above | BSD-3-Clause | |
| `dandelion` | sc-dandelion | AGPL-3.0-or-later | {mod}`spaclone.ext.dandelion` |
| `immcantation` | Change-O (with pRESTO and airr) | AGPL-3.0 (Change-O, pRESTO); CC BY 4.0 (airr) | `MakeDb.py`, `DefineClones.py`, `CreateGermlines.py` for {mod}`spaclone.ext.immcantation`; IgBLAST and R are separate programs (below) |
| `scicsr` | CellRank (2.3 or newer on Python 3.12+, 2.0 on 3.11), deeptime, pysam | BSD-3-Clause (CellRank); LGPL-3.0 (deeptime); MIT (pysam) | `backend="cellrank"` / `"deeptime"` of the transition models; BAM input of the germline-transcript counts |
| `threadfin` | Threadfin | MIT | {mod}`spaclone.ext.threadfin` |
| `tcr` | OLGA | GPL-3.0 | {func}`~spaclone.tl.pgen`; `pgen=True` in {func}`~spaclone.tl.public_clonotypes` |
| `leiden` | leidenalg | GPL-3.0-or-later | scanpy's leidenalg backend (spaclone's own clustering uses igraph) |
| `datasets` | pooch | BSD-3-Clause | downloads in {mod}`spaclone.datasets` |
| `images` | tifffile | BSD-3-Clause | {func}`~spaclone.io.read_tissue_mask` |
| `fast` | numba | BSD | nothing required (scanpy already installs it) |
| `full` | all of the above | all of the above | |

## Licences

spaclone's own code is MIT-licensed. It does not bundle the packages it works with: each one is a separate download,
installed by you, and you use it under its own licence. The optional packages and programs include copyleft ones
(GPL, LGPL, AGPL), as the tables show; the MIT licence of spaclone does not change their terms. The small format
fixtures of the test suite carry their own attributions (`tests/_data/formats/README.md`). How to cite each package:
[](citing.md).

| Core dependency | Licence |
|---|---|
| anndata, scanpy, h5py, numpy, pandas, scikit-learn, scipy | BSD-3-Clause |
| igraph (python-igraph) | GPL-2.0-or-later |
| matplotlib | Matplotlib licence (PSF-based) |
| pyarrow | Apache-2.0 |

| External program or data | Licence |
|---|---|
| IgBLAST | public domain (NCBI) |
| IgPhyML | GPL-2.0-or-later |
| R | GPL-2 or GPL-3 |
| alakazam, shazam, tigger, dowser, scoper | AGPL-3 |
| sciCSR (and its NMF signatures, which {mod}`spaclone.ext.scicsr` downloads when asked) | GPL-3 |
| IMGT germline sequences (`--references`) | IMGT's terms (CC BY-NC-ND 4.0); not shipped with spaclone |

The licences are those in each package's metadata at the versions tested (2026-10-31).

## Everything, with IgBLAST and R (conda)

IgBLAST, IgPhyML and the Immcantation R packages (SHazaM, TIgGER, dowser, SCOPer, alakazam) are not Python packages.
One script builds a conda environment with all of them and spaclone[full]:

```bash
git clone https://github.com/Reynold4k/spaclone.git
cd spaclone
bash scripts/install_spaclone_env.sh
conda activate spaclone          # or: micromamba activate spaclone
```

It uses micromamba, mamba or conda (and downloads micromamba when none is found), creates the environment from
`environment.yml` (Python 3.12; IgBLAST, Change-O, pRESTO and IgPhyML from bioconda; R 4.4 with the compiled
dependencies from conda-forge), installs spaclone[full] from the clone, installs alakazam, shazam, tigger, dowser and
scoper from CRAN (their conda builds are several releases behind) and ends with `check_install()`. Options:

| Option | |
|---|---|
| `--prefix DIR` / `--name NAME` | where the environment goes (default: a named environment `spaclone`) |
| `--python 3.11` | another Python version |
| `--references DIR` | also download the IMGT germlines and build the IgBLAST databases in `DIR` (Immcantation's `fetch_imgtdb.sh` and `imgt2igblast.sh`) and point spaclone at them |
| `--with-scicsr` | also build the original sciCSR as two separate environments (R 4.3 with Seurat 4 and sciCSR; Python 3.8 with CellRank 1.5, which sciCSR needs and which cannot share spaclone's Python) |
| `--no-r` | skip R |

`conda env create -f environment.yml` alone gives the same environment without the CRAN packages.

## External programs without the script

spaclone finds a program through an explicit argument, then an environment variable, then the folder of the running
Python (programs that pip put into this environment), then `PATH`:

| Program | Install | Variable |
|---|---|---|
| IgBLAST and its data | `conda install -c bioconda igblast`, or [NCBI](https://ncbi.github.io/igblast/) | `SPACLONE_IGBLAST` (program or folder), `SPACLONE_IGDATA` (or IgBLAST's own `IGDATA`) |
| IMGT germlines (IMGT-gapped FASTA) and IgBLAST databases | Immcantation's `fetch_imgtdb.sh` + `imgt2igblast.sh` | `SPACLONE_GERMLINES` (folder of `imgt_<species>_IGHV.fasta`, ...) |
| Change-O | `pip install "spaclone[immcantation]"` | `SPACLONE_CHANGEO` (folder) |
| R with alakazam, shazam, tigger, dowser, scoper | `install.packages(c("alakazam", "shazam", "tigger", "dowser", "scoper"))` | `SPACLONE_RSCRIPT` |
| IgPhyML | `conda install -c bioconda igphyml` | `SPACLONE_IGPHYML` |
| The original sciCSR | `bash scripts/install_spaclone_env.sh --with-scicsr` | `SPACLONE_SCICSR_RSCRIPT`, `SPACLONE_SCICSR_PYTHON` |

`check_install()` reports each of them, with its version and path.

(tested-installations)=
## Tested installations

Fresh environments built with plain `pip` from the repository on Linux (2026-10-31), each followed by `pip check`,
`check_install()` and the test suite:

| Python | Install | Resolved versions | Result |
|---|---|---|---|
| 3.11 | core | numpy 2.4, pandas 2.3, scanpy 1.11, anndata 0.12 | all tests pass |
| 3.12 | core | numpy 2.5, pandas 3.1, scanpy 1.12, anndata 0.13 | all tests pass |
| 3.13 | core | numpy 2.5, pandas 3.1, scanpy 1.12, anndata 0.13 | all tests pass |
| 3.11 | `[full]` | scirpy 0.22, squidpy 1.8.2, SpatialData 0.7, sc-dandelion 1.0.1, CellRank 2.0.7, Threadfin 4.6, Change-O 1.3.5 | every optional Python feature available; bridges tested |
| 3.12 | `[full]` | scirpy 0.25, squidpy 1.8.4, SpatialData 0.8, sc-dandelion 1.0.1, CellRank 2.3.3 (scipy 1.16), Threadfin 4.6, Change-O 1.3.5 | every optional Python feature available; bridges tested |
| 3.13 | `[full]` | as 3.12 | every optional Python feature available; bridges tested |
| 3.12 | `scripts/install_spaclone_env.sh` (conda) | the 3.12 `[full]` stack + IgBLAST 1.22, Change-O 1.3.5, IgPhyML, R 4.4 with alakazam 1.5.0, shazam 1.4.0, tigger 1.2.0, dowser 2.5.2, scoper 1.6.0 | all tests pass, including the IgBLAST / Change-O / SHazaM / dowser round trips; `--references` builds the IMGT databases |

The external programs used in these tests were IgBLAST 1.22, Change-O 1.3.5 and R 4.5 with alakazam 1.5.0, shazam
1.4.0, tigger 1.2.0, dowser 2.5.2 and scoper 1.6.0; the original sciCSR 0.3.5 ran in its own environment (R 4.3,
Seurat 4.4; Python 3.8, CellRank 1.5.1).

## When something goes wrong

- **Packages missing although pip said they were installed.** A `PYTHONPATH` set in the shell makes pip treat the
  packages on that path as installed. Run `unset PYTHONPATH` before creating the environment and before using it.
- **`broken` in `check_install()`.** The package is there but its import fails; the error is shown. sc-dandelion 0.5
  needs `matplotlib<3.9`; sc-dandelion 1.0 or newer has no such pin (its clone calls differ from 0.5, see
  [](ecosystem.md)).
- **A program found but `broken`.** It is on `PATH` but does not run, typically a script whose Python cannot see its
  packages (for example Change-O installed with `pip install --user` while `PYTHONNOUSERSITE` is set). Point the
  variable of the table above at a working installation.
- **scipy older than expected with `[scicsr]`.** CellRank 2.3 needs pygam 0.10 or newer, which caps scipy below 1.17;
  pip then installs scipy 1.16.
