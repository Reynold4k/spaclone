#!/usr/bin/env bash
# Build the complete spaclone environment in one command:
#   Python 3.12 + spaclone[full] (scirpy, squidpy, SpatialData, Dandelion, CellRank, Threadfin, Change-O, ...)
#   + IgBLAST, Change-O, pRESTO, IgPhyML (bioconda) + R with alakazam, shazam, tigger, dowser, scoper (CRAN)
# and finish with spaclone.check_install().
#
# Usage: bash scripts/install_spaclone_env.sh [options]
#   --prefix DIR        create the environment at DIR (default: a named environment, see --name)
#   --name NAME         name of the environment (default: spaclone)
#   --source SRC        spaclone to install: a local clone (default: the clone holding this script) or a pip URL
#   --python X.Y        Python version (default: 3.12; 3.11 to 3.13 are supported)
#   --frontend PROG     micromamba, mamba or conda (default: the first one found; micromamba is downloaded if none)
#   --no-r              skip R and the Immcantation R packages
#   --references DIR    also download the IMGT germlines and build the IgBLAST databases in DIR (Immcantation's
#                       fetch_imgtdb.sh and imgt2igblast.sh), and point spaclone at them
#   --with-scicsr       also build the original sciCSR (R 4.3 + Seurat 4 + sciCSR; Python 3.8 + CellRank 1.5) as two
#                       separate environments, <env>-scicsr-r and <env>-scicsr-py
#   --jobs N            parallel jobs for compiling R packages (default: 4)
#   -h, --help          this text
# Afterwards: activate the environment (conda activate spaclone / micromamba activate spaclone) or call its python.
set -euo pipefail

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREFIX="" NAME="spaclone" SOURCE="" PYVER="3.12" FRONTEND="" WITH_R=1 REFS="" SCICSR=0 JOBS=4
export JOBS
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --source) SOURCE=$2; shift 2 ;;
    --python) PYVER=$2; shift 2 ;;
    --frontend) FRONTEND=$2; shift 2 ;;
    --no-r) WITH_R=0; shift ;;
    --references) REFS=$2; shift 2 ;;
    --with-scicsr) SCICSR=1; shift ;;
    --jobs) JOBS=$2; export JOBS; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# A stray PYTHONPATH makes pip skip packages that the environment then lacks.
unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER
export PYTHONNOUSERSITE=1

# ------------------------------------------------------------------ conda frontend
if [ -z "$FRONTEND" ]; then
  for f in micromamba mamba conda; do
    if command -v "$f" >/dev/null 2>&1 && "$f" --version >/dev/null 2>&1; then FRONTEND=$(command -v "$f"); break; fi
  done
fi
if [ -z "$FRONTEND" ]; then
  BIN="${XDG_DATA_HOME:-$HOME/.local/share}/spaclone/bin"
  mkdir -p "$BIN"
  say "no conda frontend found: downloading micromamba to $BIN"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$BIN" --strip-components=1 bin/micromamba
  FRONTEND="$BIN/micromamba"
fi
FE=$(basename "$FRONTEND")
if [ "$FE" = "micromamba" ] && [ -z "${MAMBA_ROOT_PREFIX:-}" ]; then
  export MAMBA_ROOT_PREFIX="${XDG_DATA_HOME:-$HOME/.local/share}/spaclone/mamba"
fi
if [ -n "$PREFIX" ]; then TARGET=(-p "$PREFIX"); else TARGET=(-n "$NAME"); fi

env_path() {  # the prefix of an environment given by its target arguments
  if [ "$1" = "-p" ]; then echo "$2"; return; fi
  if [ "$FE" = "micromamba" ]; then echo "$MAMBA_ROOT_PREFIX/envs/$2"; else
    "$FRONTEND" env list | awk -v n="$2" '$1 == n {print $NF}'; fi
}
create() {  # create <yml> <target args...>
  local yml=$1; shift
  if [ "$FE" = "micromamba" ]; then "$FRONTEND" create -y "$@" -f "$yml"
  else "$FRONTEND" env create "$@" -f "$yml"; fi
}

# ------------------------------------------------------------------ the environment (without the pip step)
say "creating the environment with $FE (Python $PYVER)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
YML="$TMP/environment.yml"
awk -v py="$PYVER" -v r="$WITH_R" '
  /^  - pip:/ {skip=1; next}
  skip && /^      - / {next}
  {skip=0}
  /^  - python=/ {print "  - python=" py; next}
  r == 0 && /^  - (r-|bioconductor-|compilers|make|pkg-config)/ {next}
  {print}' "$HERE/environment.yml" > "$YML"
create "$YML" "${TARGET[@]}"
ENV=$(env_path "${TARGET[@]}")
[ -x "$ENV/bin/python" ] || die "environment not found at $ENV"
PY="$ENV/bin/python"

# ------------------------------------------------------------------ spaclone[full]
if [ -z "$SOURCE" ]; then
  if [ -f "$HERE/pyproject.toml" ]; then SOURCE=$HERE; else SOURCE="git+https://github.com/Reynold4k/spaclone.git"; fi
fi
say "installing spaclone[full] from $SOURCE"
if [ -d "$SOURCE" ]; then SPEC="$SOURCE[full]"; else SPEC="spaclone[full] @ $SOURCE"; fi
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "$SPEC"

# ------------------------------------------------------------------ R packages of Immcantation (CRAN)
if [ "$WITH_R" = 1 ]; then
  say "installing alakazam, shazam, tigger, dowser and scoper from CRAN"
  # the environment's compilers (x86_64-conda-linux-gnu-c++) must be on PATH for R's Makeconf
  PATH="$ENV/bin:$PATH" "$ENV/bin/Rscript" -e 'pk <- c("alakazam", "shazam", "tigger", "dowser", "scoper");
    need <- pk[!vapply(pk, requireNamespace, logical(1), quietly = TRUE)];
    if (length(need)) install.packages(need, repos = "https://cloud.r-project.org", Ncpus = as.integer(Sys.getenv("JOBS", "4")));
    for (p in pk) suppressPackageStartupMessages(library(p, character.only = TRUE));
    cat(paste(pk, vapply(pk, function(p) as.character(packageVersion(p)), ""), collapse = ", "), "\n")'
fi

# ------------------------------------------------------------------ settings read by spaclone
ACT="$ENV/etc/conda/activate.d"
mkdir -p "$ACT"
{
  echo "# written by spaclone's install_spaclone_env.sh"
  [ -d "$ENV/share/igblast" ] && echo "export SPACLONE_IGDATA=\"$ENV/share/igblast\""
} > "$ACT/spaclone.sh"

# ------------------------------------------------------------------ IMGT references (optional)
if [ -n "$REFS" ]; then
  say "downloading IMGT germlines and building IgBLAST databases in $REFS"
  mkdir -p "$REFS"
  base="https://raw.githubusercontent.com/immcantation/immcantation/master/scripts"
  for s in fetch_imgtdb.sh imgt2igblast.sh; do curl -fsSL "$base/$s" -o "$TMP/$s"; chmod +x "$TMP/$s"; done
  PATH="$ENV/bin:$PATH" "$TMP/fetch_imgtdb.sh" -o "$REFS/germlines"
  PATH="$ENV/bin:$PATH" "$TMP/imgt2igblast.sh" -i "$REFS/germlines" -o "$REFS/igblast"
  for d in internal_data optional_file; do
    [ -d "$ENV/share/igblast/$d" ] && cp -r "$ENV/share/igblast/$d" "$REFS/igblast/"
  done
  {
    echo "export SPACLONE_IGDATA=\"$REFS/igblast\""
    echo "export SPACLONE_GERMLINES=\"$REFS/germlines/imgt/mouse/vdj\"   # change mouse to your species"
  } >> "$ACT/spaclone.sh"
fi

# ------------------------------------------------------------------ the original sciCSR (optional)
if [ "$SCICSR" = 1 ]; then
  say "building the original sciCSR (two environments)"
  if [ -n "$PREFIX" ]; then RT=(-p "$PREFIX-scicsr-r"); PT=(-p "$PREFIX-scicsr-py")
  else RT=(-n "$NAME-scicsr-r"); PT=(-n "$NAME-scicsr-py"); fi
  create "$HERE/envs/scicsr-r.yml" "${RT[@]}"
  create "$HERE/envs/scicsr-py.yml" "${PT[@]}"
  RENV=$(env_path "${RT[@]}"); PENV=$(env_path "${PT[@]}")
  # SeuratObject 4.1.4 from conda was built against an older Matrix: rebuild it from source when Seurat fails
  if ! "$RENV/bin/Rscript" -e 'suppressMessages(library(Seurat)); x <- CreateSeuratObject(matrix(rpois(2000, 3), 20, 100,
      dimnames = list(paste0("g", 1:20), paste0("c", 1:100)))); x <- FindNeighbors(RunPCA(ScaleData(NormalizeData(x,
      verbose = FALSE), verbose = FALSE), features = rownames(x), npcs = 5, verbose = FALSE), dims = 1:5, verbose = FALSE)' \
      >/dev/null 2>&1; then
    curl -fsSL https://cran.r-project.org/src/contrib/Archive/SeuratObject/SeuratObject_4.1.4.tar.gz -o "$TMP/so.tar.gz"
    PATH="$RENV/bin:$PATH" "$RENV/bin/R" CMD INSTALL "$TMP/so.tar.gz"
  fi
  PATH="$RENV/bin:$PATH" "$RENV/bin/Rscript" -e 'if (!requireNamespace("sciCSR", quietly = TRUE))
      remotes::install_github("Fraternalilab/sciCSR", upgrade = "never", dependencies = FALSE);
      suppressMessages(library(sciCSR)); cat("sciCSR", as.character(packageVersion("sciCSR")), "\n")'
  "$PENV/bin/python" -s -c 'from cellrank.tl.kernels import PseudotimeKernel; import cellrank; print("cellrank", cellrank.__version__)'
  {
    echo "export SPACLONE_SCICSR_RSCRIPT=\"$RENV/bin/Rscript\""
    echo "export SPACLONE_SCICSR_PYTHON=\"$PENV/bin/python\""
  } >> "$ACT/spaclone.sh"
fi

# ------------------------------------------------------------------ check
say "checking the installation"
set +u
# shellcheck disable=SC1090
. "$ACT/spaclone.sh"
set -u
"$PY" -c 'from spaclone._versions import check_install; print(check_install())'
cat <<EOF

Done. The environment is $ENV
Use it with:   $FE activate ${PREFIX:-$NAME}     (or call $PY directly)
Check it with: python -c "import spaclone; print(spaclone.check_install())"
EOF
