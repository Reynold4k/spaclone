"""Readers that return receptor tables with micrometre coordinates (``SpatialAIR``), for every mainstream format.

``spaclone.read(gex, airr)`` reads a spatial GEX source and a receptor table of any supported tool;
``spaclone.io.sniff(path)`` says what a file is and what must still be given; ``read_receptors`` reads a receptor
table alone; ``read_gex`` a GEX source alone.
"""

from ._container import *  # noqa: F403
from .airr import *  # noqa: F403
from .airspace import *  # noqa: F403
from .annotations import *  # noqa: F403
from .barcodes import *  # noqa: F403
from .cellcalls import *  # noqa: F403
from .formats import *  # noqa: F403
from .gexio import *  # noqa: F403
from .interop import *  # noqa: F403
from .mixcr import *  # noqa: F403
from .receptors import *  # noqa: F403
from .sniff import *  # noqa: F403
from .stereo import *  # noqa: F403
from .store import *  # noqa: F403

from .paired import *  # noqa: F403  # isort: skip  (after .store: io.read reads the GEX + AIRR pair, or a store)
