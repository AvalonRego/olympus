"""Collection of classes implementing a detector."""
import itertools
from dataclasses import dataclass, field
from typing import Tuple

import awkward as ak
import numpy as np

from ananke.models.detector import Detector as AnankeDetector
from ananke.schemas.detector import DetectorConfiguration
from ananke.services.detector import DetectorBuilderService

from .utils import get_event_times_by_rate

@dataclass
class Detector(AnankeDetector):
    """Olympus's implementation of Ananke Detector."""

    #: distance from center to the most outer string
    outer_radius: float = field(init=False)

    #: radius from part one and height as second value
    outer_cylinder: Tuple[float, float] = field(init=False)


    def __post_init__(self) -> None:
        """Set additional outer radius and cylinder to class."""
        dataframe = self.to_pandas()

        module_locations = np.array(dataframe[['module_x', 'module_y', 'module_z']])

        # TODO: check whether radius correct
        self.outer_radius = np.linalg.norm(module_locations, axis=1).max()
        self.outer_cylinder = (
            np.linalg.norm(module_locations[:, :2], axis=1).max(),
            2 * np.abs(module_locations[:, 2].max()),
        )

class DetectorBuilder(DetectorBuilderService):
    """DetectorBuilder override for Olympus"""
    def __init__(self):
        super().__init__(Detector)

    def get(self, configuration: DetectorConfiguration) -> Detector:
        return super().get(configuration)



def sample_cylinder_surface(height, radius, n, rng=np.random.RandomState(1337)):
    """Sample points on a cylinder surface."""
    side_area = 2 * np.pi * radius * height
    top_area = 2 * np.pi * radius**2

    ratio = top_area / (top_area + side_area)

    is_top = rng.uniform(0, 1, size=n) < ratio
    n_is_top = is_top.sum()
    samples = np.empty((n, 3))
    theta = rng.uniform(0, 2 * np.pi, size=n)

    # top / bottom points

    r = radius * np.sqrt(rng.uniform(0, 1, size=n_is_top))

    samples[is_top, 0] = r * np.sin(theta[is_top])
    samples[is_top, 1] = r * np.cos(theta[is_top])
    samples[is_top, 2] = rng.choice(
        [-height / 2, height / 2], replace=True, size=n_is_top
    )

    # side points

    r = radius
    samples[~is_top, 0] = r * np.sin(theta[~is_top])
    samples[~is_top, 1] = r * np.cos(theta[~is_top])
    samples[~is_top, 2] = rng.uniform(-height / 2, height / 2, size=n - n_is_top)

    return samples


def sample_cylinder_volume(height, radius, n, rng=np.random.RandomState(1337)):
    """Sample points in cylinder volume."""
    theta = rng.uniform(0, 2 * np.pi, size=n)
    r = radius * np.sqrt(rng.uniform(0, 1, size=n))
    samples = np.empty((n, 3))
    samples[:, 0] = r * np.sin(theta)
    samples[:, 1] = r * np.cos(theta)
    samples[:, 2] = rng.uniform(-height / 2, height / 2, size=n)
    return samples


def sample_direction(n_samples, rng=np.random.RandomState(1337)):
    """Sample uniform directions."""
    cos_theta = rng.uniform(-1, 1, size=n_samples)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0, 2 * np.pi)

    samples = np.empty((n_samples, 3))
    samples[:, 0] = np.sin(theta) * np.cos(phi)
    samples[:, 1] = np.sin(theta) * np.sin(phi)
    samples[:, 2] = np.cos(theta)

    return samples


def get_proj_area_for_zen(height, radius, coszen):
    """Return projected area for cylinder."""
    cap = np.pi * radius * radius
    sides = 2 * radius * height
    return cap * np.abs(coszen) + sides * np.sqrt(1.0 - coszen * coszen)


def generate_noise(det, time_range, rng=np.random.RandomState(1337)):
    """Generate detector noise in a time range."""
    all_times_det = []
    dT = np.diff(time_range)
    for idom in range(len(det.modules)):
        times_det = get_event_times_by_rate(det.modules[idom].noise_rate, time_range[0], time_range[1], rng=rng)
        all_times_det.append(times_det)

    return ak.sort(ak.Array(all_times_det))


def trigger(det, event_times, mod_thresh=8, phot_thres=5):
    """
    Check a simple multiplicity condition.

    Trigger is true when at least `mod_thresh` modules have measured more than `phot_thres` photons.

    Parameters:
        det: Detector
        event_times: ak.array
        mod_thresh: int
            Threshold for the number of modules which have detected `phot_thres` photons
        phot_thres: int
            Threshold for the number of photons per module
    """
    hits_per_module = ak.count(event_times, axis=1)
    if ak.sum((hits_per_module > phot_thres)) > mod_thresh:
        return True
    return False


"""
def local_coinc(hit_times, lc_links, pmt_t=50, lc_t=500, smt_t=1000):

    trigger_times = []
    mod_ids = []
    lc_c
    for mid in range(len(hit_times)):
        ts_l = ak.sort(ak.flatten(hit_times[lc_links[mid]]))
        ts_mod = hit_times[mid]

        # More than two hits within 50 ns
        valid = (ts_mod[1:] - ts_mod[:-1]) < pmt_t

        triggers = np.zeros(ak.sum(valid), dtype=np.bool)
        for i, vhit in enumerate(ts_mod[valid]):

            # At least one hit within 500ns on neighboring module
            if np.any(np.abs(ts_l - vhit) < lc_t):
                triggers[i] = True
        trigger_times.append(ts_mod[valid][triggers])
        mod_ids.append(np.ones(triggers.shape[0]) * mid)

    trigger_times = ak.concatenate(trigger_times)
    return ak.sum((trigger_times[1:] - trigger_times[:-1]) < smt_t)
"""
