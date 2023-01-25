"""Module containing code for mock photon source propagation."""
from multiprocessing import Pool
from typing import Union, Tuple

import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
import pandas as pd
import logging

from ananke.models.detector import Detector
from ananke.models.event import Sources, Hits, EventRecords
from olympus.configuration.photon_propagation import MockPhotonPropagatorConfiguration
from olympus.event_generation.lightyield import fennel_angle_distribution_function
from olympus.event_generation.medium import Medium
from olympus.event_generation.photon_propagation.interface import \
    AbstractPhotonPropagator


def photons_per_pmt_to_hits(
        record_id: int,
        record_type: str,
        pmt_photons: jnp.ndarray,
        pmt_times: jnp.ndarray,
        angular_yield: jnp.ndarray,
        pmt_indices: Tuple[int, int, int],
        rng: np.random.BitGenerator
) -> Hits:
    """Transforms all the sources of one PMT to hits for that PMT.

    Args:
        record_id: ID of the used record
        record_type: Type of the used record
        pmt_photons: List with number of photons per source
        pmt_times: List with times per source and photons
        angular_yield: Yield of sources dependent on angle for gamma distribution
        pmt_indices: Indices of the current PMT
        rng: Random number generator

    Returns:
        All hits for the current PMT
    """
    logging.log(
        logging.INFO,
        "Processed PMT {} for Record {}".format(
            pmt_indices, record_id
        )
    )

    # TODO: Verify the distributions look nice
    scaled_angular = angular_yield * 1E4
    gamma = 2 * jnp.exp(-0.5 * scaled_angular)

    hit_times = []

    for index, photons in enumerate(pmt_photons):
        if photons == 0:
            continue
        # TODO: Check passing of RNG generator
        # TODO: Flatten everything
        distribution = rng.gamma(
            gamma[index],
            gamma[index],
            photons
        ) + pmt_times[index]
        hit_times.append(distribution)

    records_hits_df = pd.DataFrame(
        {
            'time': jnp.concatenate(hit_times),
            'string_id': pmt_indices[0],
            'module_id': pmt_indices[1],
            'pmt_id': pmt_indices[2],
            'record_id': record_id,
            'type': record_type,
        }
    )

    return Hits(df=records_hits_df)


def unit_vector(
        vector: Union[jnp.ndarray, np.typing.NDArray, Tuple[float, float, float]],
        axis: Union[int, Tuple[int, int]] = 0
) -> jnp.ndarray:
    """ Returns the unit vector of the vector.

    Args:
        vector: vector to calculate unit vector of.
        axis: axis of the operation on input.

    Returns:
        Unit vector in direction of input vector.

    """
    norm = jnp.expand_dims(jnp.linalg.norm(vector, axis=axis), axis=axis)
    # TODO: Investigate Bug with jnp division precision
    return jnp.array(
        np.divide(
            vector,
            norm,
            out=np.zeros_like(vector),
            where=norm != 0
        )
    )


def angle_between(
        v1: Union[jnp.ndarray, np.typing.NDArray],
        v2: Union[jnp.ndarray, np.typing.NDArray],
        axis: Tuple[int, int]
):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793

    Args:
        v1: First vector
        v2: Second vector
        axis: axis by which to calculate

    Returns:
        Angle between vectors in Radians
    """
    v1_u = unit_vector(v1, axis=axis[0])
    v2_u = unit_vector(v2, axis=axis[1])

    return jnp.arccos(jnp.clip(jnp.einsum('...i, ...i', v1_u, v2_u), -1.0, 1.0))


class MockPhotonPropagator(AbstractPhotonPropagator[MockPhotonPropagatorConfiguration]):
    """Class enabling basic photon propagation to detector modules."""

    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # TODO: Migrate Angle and Wavelength away
        self.resolution = self.configuration.resolution

        pmt_locations = self.detector.pmt_locations.to_numpy(np.float32)
        self.pmt_positions = jnp.array(pmt_locations)

        self.pmt_indices = self.detector.indices

        pmt_orientations = self.detector.pmt_orientations.to_numpy(np.float32)
        self.pmt_orientations = jnp.array(pmt_orientations)

        pmt_areas = self.detector.pmt_areas.to_numpy(np.float32)
        self.pmt_areas = jnp.array(pmt_areas)
        self.pmt_radius = jnp.sqrt(jnp.array(pmt_areas) / jnp.pi)

        pmt_efficiencies = self.detector.pmt_efficiencies.to_numpy(np.float32)
        self.pmt_efficiencies = jnp.array(pmt_efficiencies)

        module_radius = self.detector.module_radius.to_numpy(np.float32)
        self.module_radius = jnp.array(module_radius)

    def __calculate_orthogonals(self, source_orientations: jnp.ndarray) -> jnp.ndarray:
        orthogonal = jnp.cross(
            jnp.expand_dims(self.pmt_orientations, axis=1),
            source_orientations
        )
        return jnp.cross(source_orientations, orthogonal)

    @staticmethod
    def __arccos(dot_products: jnp.ndarray) -> jnp.ndarray:
        return jnp.arccos(jnp.clip(dot_products, -1.0, 1.0))

    def __calculate_pmt_to_source_angles(
            self,
            pmt_to_source: jnp.ndarray
    ) -> jnp.ndarray:
        per_source_and_pmt_unit = unit_vector(pmt_to_source, axis=2)
        per_pmt_unit = unit_vector(self.pmt_orientations, axis=1)
        dot_products = jnp.einsum(
            'ijk,ik->ij',
            per_source_and_pmt_unit,
            per_pmt_unit
        )

        return self.__arccos(dot_products=dot_products)

    @classmethod
    def __calculate_source_orientation_angles(
            cls,
            pmt_to_source: jnp.ndarray,
            source_orientations: jnp.ndarray
    ) -> jnp.ndarray:
        per_source_and_pmt_unit = unit_vector(pmt_to_source, axis=2)
        source_orientation_unit = unit_vector(source_orientations, axis=1)
        dot_products = jnp.einsum(
            'ijk,jk->ij',
            per_source_and_pmt_unit,
            source_orientation_unit
        )

        return cls.__arccos(dot_products=dot_products)

    def __get_ellipsis_mask(
            self,
            pmt_to_source: jnp.ndarray,
            pmt_to_source_angles: jnp.ndarray,
            pmt_opening_angles_length: int,
            max_opening_angles_length: int,
            source_orientations: jnp.ndarray
    ) -> jnp.ndarray:
        """Calculates the mask of the ellipsis for each pmt and source.

        Args:
            pmt_to_source: Vectors from PMTs to Sources
            pmt_to_source_angles: Angles between PMTs and Sources
            pmt_opening_angles_length: Number of indices to account for of PMT
            max_opening_angles_length: Ceiled number of PMT opening angles
            source_orientations: Vectors representing source orientations

        Returns:
            per source mask for the PMTs elliptical field of view
        """
        # calculate rotation of ellipsis
        # first find pmt normal projection on target area
        unit_pmt_to_source = unit_vector(pmt_to_source, axis=2)
        pmt_normal_to_source_normal_distance = jnp.einsum(
            'ijk,ik->ij',
            unit_pmt_to_source,
            self.pmt_orientations
        )
        projected_pmt_normal = jnp.subtract(
            jnp.expand_dims(self.pmt_orientations, axis=1),
            jnp.einsum(
                'ij,ijk->ijk',
                pmt_normal_to_source_normal_distance,
                unit_pmt_to_source
            )
        )
        # second get second vector of target area basis orthogonal to source direction
        second_target_area_basis = jnp.cross(source_orientations, pmt_to_source)
        # third calculate ellipsis properties
        ellipsis_a_axis = pmt_opening_angles_length / 2
        ellipsis_b_axis = jnp.einsum(
            'ij,ij->ij',
            jnp.cos(pmt_to_source_angles),
            ellipsis_a_axis
        )
        ellipsis_angles = angle_between(
            projected_pmt_normal,
            second_target_area_basis,
            axis=(2, 2)
        )

        center_index = int(jnp.floor(max_opening_angles_length / 2))
        target_indices = jnp.arange(max_opening_angles_length) \
                             .astype('int16') - center_index
        target_full = jnp.einsum('i,j->ij', target_indices, target_indices)

        sin_angles = jnp.sin(ellipsis_angles)
        cos_angles = jnp.cos(ellipsis_angles)

        squared_target = jnp.square(target_indices)
        first_ellipsis_new = jnp.einsum(
            'ij,k->ijk',
            jnp.add(
                jnp.square(jnp.divide(sin_angles, ellipsis_a_axis)),
                jnp.square(jnp.divide(cos_angles, ellipsis_b_axis))
            ),
            squared_target
        )

        last_ellipsis_new = jnp.einsum(
            'ij,k->ijk',
            jnp.add(
                jnp.square(jnp.divide(cos_angles, ellipsis_a_axis)),
                jnp.square(jnp.divide(sin_angles, ellipsis_b_axis))
            ),
            squared_target
        )

        middle_multiply = jnp.einsum(
            'ij,ij->ij',
            jnp.einsum('ij,ij->ij', sin_angles, cos_angles),
            jnp.subtract(
                jnp.reciprocal(jnp.square(ellipsis_a_axis)),
                jnp.reciprocal(jnp.square(ellipsis_b_axis))
            )
        )

        # TODO: Check Ellipsis Axis
        return (
                       jnp.expand_dims(first_ellipsis_new, axis=2) +
                       2 * jnp.einsum('ij,kl->ijkl', middle_multiply, target_full) +
                       jnp.expand_dims(last_ellipsis_new, axis=3)
               ) <= 1

    def __get_ellipsis_yield(
            self,
            source_angle_distribution: jnp.ndarray,
            number_of_modules: int,
            angle_indices: jnp.ndarray,
            pmt_to_source: jnp.ndarray,
            pmt_to_source_angles: jnp.ndarray,
            pmt_opening_angles_length: int,
            max_opening_angles_length: int,
            source_orientations: jnp.ndarray
    ) -> jnp.ndarray:
        """Calculates the mask of the ellipsis for each pmt and source.

        Args:
            source_angle_distribution: distribution of the angles of the sources
            number_of_modules: number of PMTs
            angle_indices: indices of the interesting angles
            pmt_to_source: Vectors from PMTs to Sources
            pmt_to_source_angles: Angles between PMTs and Sources
            pmt_opening_angles_length: Number of indices to account for of PMT
            max_opening_angles_length: Ceiled number of PMT opening angles
            source_orientations: Vectors representing source orientations

        Returns:
            per source and pmt yield in percentage
        """
        angular_distribution_per_pmt = jnp.take(
            jnp.tile(source_angle_distribution, (number_of_modules, 1)),
            angle_indices
        )
        ellipsis_mask = self.__get_ellipsis_mask(
            source_orientations=source_orientations,
            pmt_to_source=pmt_to_source,
            pmt_to_source_angles=pmt_to_source_angles,
            pmt_opening_angles_length=pmt_opening_angles_length,
            max_opening_angles_length=max_opening_angles_length
        )

        return jnp.einsum(
            '...ij,...i->...',
            ellipsis_mask,
            angular_distribution_per_pmt
        )

    def __calculate_angular_yield(
            self,
            number_of_modules: int,
            number_of_sources: int,
            source_orientations: jnp.ndarray,
            source_angle_distribution: jnp.ndarray,
            pmt_to_source: jnp.ndarray,
            pmt_to_source_distances: jnp.ndarray,
    ) -> jnp.ndarray:
        pmt_to_source_angles = self.__calculate_pmt_to_source_angles(pmt_to_source)
        source_orientation_angles = self.__calculate_source_orientation_angles(
            - pmt_to_source,  # we want to know angle between source and the pmt.
            source_orientations
        )
        pmt_opening_angles = 2 * jnp.arcsin(
            jnp.divide(
                self.pmt_radius[:, jnp.newaxis],
                pmt_to_source_distances
            )
        )
        pmt_opening_angles_length = pmt_opening_angles / jnp.pi * self.resolution
        source_orientation_angles_index = jnp.rint(
            source_orientation_angles / jnp.pi * self.resolution
        ).astype('int16')
        # this is the maximum size of the arrays to consider
        max_opening_angle_length = int(jnp.ceil(jnp.max(pmt_opening_angles_length)))
        angle_indices = jnp.indices((max_opening_angle_length,)) - \
                        jnp.rint(max_opening_angle_length / 2)
        angle_indices = jnp.add(
            jnp.tile(angle_indices, (number_of_modules, number_of_sources, 1)),
            jnp.expand_dims(source_orientation_angles_index, axis=2)
        ).astype('int16')
        # Get Angular distribution values for each pmt

        yield_per_source_and_pmt = self.__get_ellipsis_yield(
            source_angle_distribution=source_angle_distribution,
            number_of_modules=number_of_modules,
            angle_indices=angle_indices,
            source_orientations=source_orientations,
            pmt_to_source=pmt_to_source,
            pmt_to_source_angles=pmt_to_source_angles,
            pmt_opening_angles_length=pmt_opening_angles_length,
            max_opening_angles_length=max_opening_angle_length

        )
        # TODO: Check Integral

        # we only have yield at pmts facing the source
        mask_condition = (pmt_to_source_angles < jnp.pi / 2).astype('int16')

        yield_per_source_and_pmt = jnp.einsum(
            'ij, ij->ij',
            mask_condition,
            yield_per_source_and_pmt
        )

        return yield_per_source_and_pmt

    def __calculate_distance_yield(
            self,
            pmt_to_source_distances: jnp.ndarray
    ) -> jnp.ndarray:
        absorption_length = self.medium.get_absolute_length(
            self.default_wavelengths
        )

        return jnp.exp(
            -1.0 * jnp.divide(pmt_to_source_distances, absorption_length)
        )

    def _get_angle_distribution(
            self, energy: float, particle_id: int
    ) -> jnp.ndarray:

        angles = jnp.linspace(0, 180, self.resolution)
        n_photon = self.medium.get_refractive_index(self.default_wavelengths)
        angle_distribution = fennel_angle_distribution_function(
            energy=energy,
            particle_id=particle_id
        )
        source_angle_distribution = angle_distribution(
            angles,
            n_photon
        )

        source_angle_distribution = jnp.array(source_angle_distribution)

        # norm all angle distributions according to a cylinder
        normation_factor = jnp.sum(source_angle_distribution)
        normation_factor = normation_factor * self.resolution

        source_angle_distribution = jnp.divide(
            source_angle_distribution,
            normation_factor
        )

        return source_angle_distribution

    def propagate(
            self,
            records: EventRecords,
            sources: Sources,
            use_multiprocessing: bool = False
    ) -> Union[Hits, Tuple[Hits, jnp.array]]:
        if len(sources) == 0:
            return Hits()
        hits_list = []
        number_of_records = len(records)
        logging.info(
            'Starting to propagate {} sources from {} records.'.format(
                len(sources),
                number_of_records
            )
        )
        for record_index, record in records.df.iterrows():
            logging.info(
                'Starting with record {} of {} records.'.format(
                    record_index + 1,
                    number_of_records
                )
            )
            record_sources = sources.get_by_record(record.record_id)
            number_of_sources = len(record_sources)
            number_of_pmts = len(self.detector)

            if number_of_sources == 0:
                continue

            source_locations = record_sources.locations.to_numpy(np.float32)
            source_locations = jnp.array(source_locations)

            source_orientations = record_sources.orientations.to_numpy(np.float32)
            source_orientations = jnp.array(source_orientations)

            source_photons = record_sources.number_of_photons.to_numpy(np.float32)
            source_photons = jnp.array(source_photons)

            source_times = record_sources.times.to_numpy(np.float32)
            source_times = jnp.array(source_times)

            # 1. Calculate angle between PMT and Source direction
            expanded_pmt_positions = jnp.tile(
                jnp.expand_dims(self.pmt_positions, axis=1),
                (1, number_of_sources, 1)
            )
            pmt_to_source = source_locations - expanded_pmt_positions
            pmt_to_source_distances = jnp.linalg.norm(pmt_to_source, axis=2)

            source_angle_distribution = self._get_angle_distribution(
                energy=record.energy,
                particle_id=record.particle_id
            )

            # 2. Mask sources in direction of PMT (180°)
            angular_yield = self.__calculate_angular_yield(
                number_of_modules=number_of_pmts,
                number_of_sources=number_of_sources,
                source_orientations=source_orientations,
                pmt_to_source=pmt_to_source,
                source_angle_distribution=source_angle_distribution,
                pmt_to_source_distances=pmt_to_source_distances
            )
            # 3. Calculate Yield based on distance
            distance_yield = self.__calculate_distance_yield(
                pmt_to_source_distances=pmt_to_source_distances
            )

            # 4. Calculate Yield based on PMT Efficiency
            efficiency_yield = self.pmt_efficiencies

            # 4. Calculate Number of Photons
            total_yield = jnp.multiply(angular_yield, distance_yield)
            jnp.einsum('ij,i->ij', total_yield, efficiency_yield)
            photons_per_pmt_per_source = jnp.multiply(total_yield, source_photons.T)
            photons_per_pmt_per_source = jnp.rint(
                photons_per_pmt_per_source
            ).astype(
                'int16'
            )

            # 5. Calculate Arrival Time
            travel_time = pmt_to_source_distances / self.medium.get_c_medium_photons(
                self.default_wavelengths
            )
            times_per_pmt_per_source = jnp.add(travel_time, source_times.T)

            # 5. Distribute Yield using Gamma distribution

            multiprocessing_args = []

            for pmt_index, pmt_photons in enumerate(photons_per_pmt_per_source):
                if jnp.max(pmt_photons) > 0:
                    multiprocessing_args.append(
                        (
                            record.record_id,
                            record.type,
                            pmt_photons,
                            times_per_pmt_per_source[pmt_index],
                            angular_yield[pmt_index],
                            self.pmt_indices.iloc[pmt_index],
                            self.rng
                        )
                    )

            if not use_multiprocessing:
                for args in multiprocessing_args:
                    hits_list.append(photons_per_pmt_to_hits(*args))
            else:
                with Pool() as pool:
                    hits_list += pool.starmap(
                        photons_per_pmt_to_hits,
                        multiprocessing_args
                    )

        all_hits = Hits.concat(hits_list)

        return all_hits
