"""Read/write of mock data and results.

The common ``Datacube`` container (``kino.io.base``) plus readers/generators
that convert each data source into it (``kino.io.kinms``, ``kino.io.manga``)
and the test-data downloader (``kino.io.download``) -- see ``TEST_DATA.md``.
``kino.io.fitsio``/``kino.io.hdf5`` are FITS/HDF5 I/O for KINO's own mocks
and run provenance (README.md Phase 4), with the same four function names
on each backend.
"""

from kino.io.base import Datacube
from kino.io.fitsio import read_datacube as read_datacube_fits
from kino.io.fitsio import read_slit_observation as read_slit_observation_fits
from kino.io.fitsio import write_datacube as write_datacube_fits
from kino.io.fitsio import write_slit_observation as write_slit_observation_fits
from kino.io.hdf5 import read_datacube as read_datacube_hdf5
from kino.io.hdf5 import read_slit_observation as read_slit_observation_hdf5
from kino.io.hdf5 import write_datacube as write_datacube_hdf5
from kino.io.hdf5 import write_slit_observation as write_slit_observation_hdf5

__all__ = [
    "Datacube",
    "read_datacube_fits",
    "read_datacube_hdf5",
    "read_slit_observation_fits",
    "read_slit_observation_hdf5",
    "write_datacube_fits",
    "write_datacube_hdf5",
    "write_slit_observation_fits",
    "write_slit_observation_hdf5",
]
