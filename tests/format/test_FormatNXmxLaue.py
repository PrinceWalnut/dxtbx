"""Tests for dxtbx.format.FormatNXmxLaue."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dxtbx.format.FormatNXmxLaue import FormatNXmxLaue


@pytest.fixture
def nxmx_laue_on_disk(tmp_path, nxmx_example):
    """A copy of the shared NXmx example, on disk, with an actual scan of
    images so that the number-of-images heuristic in _is_laue_nxmx has
    something to compare against."""
    with nxmx_example as f, h5py.File(tmp_path / "test.h5", "w") as g:
        f.copy("entry", g)

        pixel_mask = "/entry/instrument/detector/pixel_mask"
        g.create_dataset(pixel_mask, data=np.zeros((100, 100), dtype=np.int32))

        # entry/data needs an actual signal dataset for the n_images
        # comparison; the shared fixture's omega array has 10 points.
        n_images = 10
        data = g["/entry/data"].create_dataset(
            "data", data=np.zeros((n_images, 100, 100))
        )
        g["/entry/data"].attrs["signal"] = "data"
        del data

        yield g


def _set_incident_wavelength(handle, values, peak_wavelength=None):
    beam = handle["/entry/instrument/beam"]
    del beam["incident_wavelength"]
    wl = beam.create_dataset("incident_wavelength", data=np.asarray(values))
    wl.attrs["units"] = b"angstrom"
    if peak_wavelength is not None:
        pw = beam.create_dataset("peak_wavelength", data=peak_wavelength)
        pw.attrs["units"] = b"angstrom"


def test_understands_wavelength_range_with_peak_wavelength(nxmx_laue_on_disk):
    with nxmx_laue_on_disk as g:
        _set_incident_wavelength(g, [0.97, 1.15], peak_wavelength=1.04)
        filename = g.filename
    assert FormatNXmxLaue.understand(filename)


def test_rejects_monochromatic_scalar_wavelength(nxmx_laue_on_disk):
    with nxmx_laue_on_disk as g:
        _set_incident_wavelength(g, 1.04)
        filename = g.filename
    assert not FormatNXmxLaue.understand(filename)


def test_rejects_shot_varying_monochromatic_wavelength(nxmx_laue_on_disk):
    """One wavelength per image (length == n_images, no peak_wavelength) is
    shot-varying monochromatic data, handled by the parent format."""
    with nxmx_laue_on_disk as g:
        _set_incident_wavelength(g, np.linspace(1.0, 1.01, 10))
        filename = g.filename
    assert not FormatNXmxLaue.understand(filename)


def test_rejects_weighted_spectrum(nxmx_laue_on_disk):
    """incident_wavelength_weights present means Beam + Spectrum instead."""
    with nxmx_laue_on_disk as g:
        _set_incident_wavelength(g, [0.97, 1.15], peak_wavelength=1.04)
        beam = g["/entry/instrument/beam"]
        beam.create_dataset("incident_wavelength_weights", data=[0.5, 0.5])
        filename = g.filename
    assert not FormatNXmxLaue.understand(filename)


def test_two_image_scan_with_peak_wavelength_is_laue(tmp_path, nxmx_example):
    """Regression test for the ambiguous case where the number of images
    equals the length (2) of the wavelength range array: without a
    peak_wavelength signal, this collides with 2-image shot-varying
    monochromatic data. peak_wavelength disambiguates it."""
    with nxmx_example as f, h5py.File(tmp_path / "two_image.h5", "w") as g:
        f.copy("entry", g)
        g.create_dataset(
            "/entry/instrument/detector/pixel_mask",
            data=np.zeros((100, 100), dtype=np.int32),
        )
        n_images = 2
        g["/entry/data"].create_dataset("data", data=np.zeros((n_images, 100, 100)))
        g["/entry/data"].attrs["signal"] = "data"

        _set_incident_wavelength(g, [0.97, 1.15], peak_wavelength=1.04)
        filename = g.filename

    assert FormatNXmxLaue.understand(filename)


def test_two_image_scan_without_peak_wavelength_is_ambiguous(tmp_path, nxmx_example):
    """Documents the inherent limitation: a 2-image scan with a 2-element
    wavelength array and no peak_wavelength cannot be distinguished from
    shot-varying monochromatic data, and is treated as the latter."""
    with nxmx_example as f, h5py.File(tmp_path / "two_image.h5", "w") as g:
        f.copy("entry", g)
        g.create_dataset(
            "/entry/instrument/detector/pixel_mask",
            data=np.zeros((100, 100), dtype=np.int32),
        )
        n_images = 2
        g["/entry/data"].create_dataset("data", data=np.zeros((n_images, 100, 100)))
        g["/entry/data"].attrs["signal"] = "data"

        _set_incident_wavelength(g, [0.97, 1.15])
        filename = g.filename

    assert not FormatNXmxLaue.understand(filename)


def test_three_image_scan_wavelength_range_without_peak_wavelength(
    tmp_path, nxmx_example
):
    """With more images than wavelength values, the range is still
    recognised even without peak_wavelength."""
    with nxmx_example as f, h5py.File(tmp_path / "three_image.h5", "w") as g:
        f.copy("entry", g)
        g.create_dataset(
            "/entry/instrument/detector/pixel_mask",
            data=np.zeros((100, 100), dtype=np.int32),
        )
        n_images = 3
        g["/entry/data"].create_dataset("data", data=np.zeros((n_images, 100, 100)))
        g["/entry/data"].attrs["signal"] = "data"

        _set_incident_wavelength(g, [0.97, 1.15])
        filename = g.filename

    assert FormatNXmxLaue.understand(filename)
