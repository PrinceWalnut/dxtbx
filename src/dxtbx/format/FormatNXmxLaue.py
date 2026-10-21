from __future__ import annotations

import h5py
import numpy as np
import nxmx

import dxtbx.nexus
from dxtbx.format.FormatNXmx import FormatNXmx, mask_cache


class FormatNXmxLaue(FormatNXmx):
    """Read NXmx-flavour NeXus data from Laue experiments.

    Recognised when incident_wavelength in the NXbeam is a two-element array
    (i.e. a min/max wavelength range rather than a single value or per-image
    value) with no associated weights. In that case the beam is modelled as a
    PolychromaticBeam spanning the min/max of the stored wavelengths, with
    peak_wavelength (if present) also recorded on the beam model.

    Files where incident_wavelength_weights (or a weighted variant) are present
    are handled by the parent FormatNXmx class instead, which builds a Beam +
    Spectrum.

    A two-element incident_wavelength array is ambiguous with shot-varying
    monochromatic data for a 2-image scan; this is resolved by checking for
    peak_wavelength, which is only meaningful for a genuine wavelength range
    (see _is_laue_nxmx).
    """

    @staticmethod
    def understand(image_file: str) -> bool:
        # Must first satisfy the base NXmx check
        if not FormatNXmx.understand(image_file):
            return False
        try:
            with h5py.File(image_file, "r") as handle:
                return FormatNXmxLaue._is_laue_nxmx(handle)
        except Exception:
            return False

    @staticmethod
    def _has_weights(beam_handle: h5py.Group) -> bool:
        """Return True if incident_wavelength has associated weights, either
        directly or via the variant chain."""
        primary_key = "incident_wavelength"
        if beam_handle.get(primary_key + "_weights") is not None:
            return True
        if beam_handle.get(primary_key + "_weight") is not None:
            return True
        # Walk the variant chain
        node = beam_handle[primary_key]
        while "variant" in node.attrs:
            variant_key = node.attrs["variant"]
            variant = beam_handle[variant_key]
            if beam_handle.get(variant_key + "_weights") is not None:
                return True
            if beam_handle.get(variant_key + "_weight") is not None:
                return True
            node = variant
        return False

    @staticmethod
    def _is_laue_nxmx(handle: h5py.File) -> bool:
        nxmx_obj = nxmx.NXmx(handle)
        if not nxmx_obj.entries:
            return False
        nxentry = nxmx_obj.entries[0]
        if not nxentry.instruments:
            return False
        nxinstrument = nxentry.instruments[0]
        if not nxinstrument.beams:
            return False
        beam_handle = nxinstrument.beams[0]._handle

        if "incident_wavelength" not in beam_handle:
            return False

        wavelength = beam_handle["incident_wavelength"]
        wl_values = np.atleast_1d(wavelength[()])

        # Scalar or single-value: monochromatic, not Laue
        if wl_values.size <= 1:
            return False

        # If weights are present, FormatNXmx handles it as Beam + Spectrum
        if FormatNXmxLaue._has_weights(beam_handle):
            return False

        # A Laue wavelength range is always exactly two values (min, max);
        # anything else without weights is shot-varying monochromatic data.
        if wl_values.size != 2:
            return False

        # An explicit peak_wavelength is only meaningful for a polychromatic
        # band (see NXmx.nxdl.xml), so its presence unambiguously identifies
        # this as a Laue range. This check must come before the n_images
        # comparison below: without it, a 2-image scan with a genuine
        # 2-element wavelength range would be indistinguishable from
        # 2-image shot-varying monochromatic data (both have array length 2
        # equal to n_images), and would be misclassified as monochromatic.
        if "peak_wavelength" in beam_handle:
            return True

        # Otherwise, fall back to comparing against the number of images:
        # a 2-element wavelength array whose length happens to equal the
        # number of images could still be shot-varying monochromatic data
        # rather than a range. This is an inherent ambiguity in the format
        # for 2-image datasets lacking peak_wavelength, which we cannot
        # resolve any further.
        try:
            nxdata = nxentry.data[0]
            if nxdata.signal:
                data = nxdata[nxdata.signal]
            else:
                data = list(nxdata.values())[0]
            n_images = data.shape[0]
        except Exception:
            n_images = None

        if n_images is not None and wl_values.size == n_images:
            # One wavelength per image — shot-varying monochromatic
            return False

        return True

    def _start(self):
        self._static_mask = None
        self._cached_file_handle = h5py.File(self._image_file, swmr=True)
        nxmx_obj = self._get_nxmx(self._cached_file_handle)
        nxentry = nxmx_obj.entries[0]
        nxsample = nxentry.samples[0]
        nxinstrument = nxentry.instruments[0]
        nxdetector = nxinstrument.detectors[0]
        nxbeam = nxinstrument.beams[0]
        nxdata = nxentry.data[0]

        self._goniometer_model = dxtbx.nexus.get_dxtbx_goniometer(nxsample)
        self._beam_factory = dxtbx.nexus.CachedPolychromaticBeamFactory(nxbeam)

        # get_dxtbx_detector needs a single wavelength for parallax correction;
        # use the beam's peak wavelength where available, else the midpoint
        # of the wavelength range.
        wavelength = self._beam_factory.get_representative_wavelength()
        self._detector_model = dxtbx.nexus.get_dxtbx_detector(
            nxdetector, wavelength, nxdata
        )
        self._scan_model = dxtbx.nexus.get_dxtbx_scan(nxsample, nxdetector)
        self._static_mask = mask_cache.store_unique_and_get(
            dxtbx.nexus.get_static_mask(nxdetector)
        )
        self._bit_depth_readout = nxdetector.bit_depth_readout

        if self._scan_model:
            self._num_images = len(self._scan_model)
        else:
            if nxdata.signal:
                data = nxdata[nxdata.signal]
            else:
                data = list(nxdata.values())[0]
            self._num_images, *_ = data.shape

    def _beam(self, index: int | None = None) -> dxtbx.model.PolychromaticBeam:
        return self._beam_factory.make_beam(index=index or 0)

    def get_spectrum(self, index=None):
        return None
