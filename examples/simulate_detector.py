import os

from ananke.configurations.detector import DetectorConfiguration
from ananke.services.detector import DetectorBuilderService
from olympus.event_generation.generators import CascadeEventGenerator
from olympus.event_generation.medium import MediumEstimationVariant, Medium
from olympus.event_generation.photon_propagation.mock_photons import MockPhotonPropagator

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

oms_per_line = 20
dist_z = 50  # m
dark_noise_rate = 16 * 1e-5  # 1/ns
side_len = 100  # m
pmts_per_module = 16
pmt_cath_area_r = 75e-3 / 2  # m
module_radius = 0.21  # m

# Calculate the relative area covered by PMTs
# efficiency = (
#         pmts_per_module * pmt_cath_area_r ** 2 * np.pi / (
#             4 * np.pi * module_radius ** 2)
# )
efficiency = 0.42 # Christian S. Number
detector_configuration = DetectorConfiguration.parse_obj(
    {
        "string": {
            "module_number": 2,
            "module_distance": 50
        },
        "pmt": {
            "efficiency": efficiency,
            "noise_rate": dark_noise_rate,
            "area": pmt_cath_area_r
        },
        "module": {
            "radius": module_radius
        },
        "geometry": {
            "type": "single",
            "side_length": 100,
        },
        "seed": 31338
    }
)

detector_service = DetectorBuilderService()
det = detector_service.get(configuration=detector_configuration)
medium = Medium(MediumEstimationVariant.PONE_OPTIMISTIC)

# photon_propagator = NormalFlowPhotonPropagator(
#     detector=det,
#     shape_model_path="../../hyperion/data/photon_arrival_time_nflow_params.pickle",
#     counts_model_path="../../hyperion/data/photon_arrival_time_counts_params.pickle",
#     medium=medium
# )
photon_propagator = MockPhotonPropagator(
    detector=det,
    medium=medium,
    angle_resolution=18000,
)

cascade_generator = CascadeEventGenerator(
    detector=det,
    particle_id=11,
    log_minimal_energy=2,
    log_maximal_energy=5,
    photon_propagator=photon_propagator
)

collection = cascade_generator.generate(1)

collection.hits.df.head()

#pickle.dump(event_collection, open('./dataset/test', "wb"))
