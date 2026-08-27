"""Sanity check that the kino package skeleton imports cleanly."""

import kino
import kino.analysis
import kino.galaxy
import kino.inference
import kino.instruments
import kino.io
import kino.lensing
import kino.mocks


def test_version():
    assert kino.__version__
