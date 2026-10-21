"""Tests for kino.galaxy.protocol.GalaxyModel."""

from __future__ import annotations

from kino.galaxy import DiscGalaxy, GalaxyModel


class TestGalaxyModelProtocol:
    def test_disc_galaxy_satisfies_the_protocol(self):
        gal = DiscGalaxy(r_e=0.5, inc=55.0, pa=30.0, v_flat=180.0)
        assert isinstance(gal, GalaxyModel)

    def test_an_unrelated_object_does_not(self):
        class NotAGalaxy:
            pass

        assert not isinstance(NotAGalaxy(), GalaxyModel)

    def test_an_object_missing_one_method_does_not(self):
        class AlmostAGalaxy:
            x0 = 0.0
            y0 = 0.0

            def surface_brightness(self, x, y):
                return x

            def vlos(self, x, y):
                return x

            # no velocity_dispersion

        assert not isinstance(AlmostAGalaxy(), GalaxyModel)
