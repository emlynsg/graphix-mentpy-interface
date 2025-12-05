"""Graphix interface for the MentPy package.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).

ref: MentPy: A Python package for parametrized MBQC circuits
Mantilla Calderón, Luis
https://github.com/mentpy/mentpy
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from graphix.fundamentals import Plane
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.parameter import Expression, Placeholder
from graphix.pauli import Pauli

import mentpy as mp

if TYPE_CHECKING:
    from graphix import Pattern


def graphix_pattern_to_mentpy(pattern: Pattern) -> mp.MBQCircuit:  # noqa: D417
    """Convert a Graphix pattern to a MentPy MBQCircuit.

    Parameters
    ----------
    pattern: graphix.Pattern

    Returns
    -------
    result: mentpy.MBQCircuit

    Exceptions
    ---------
    NotImplementedError
        If the pattern has Expression measurements not supported by MentPy
    ValueError
        If the pattern has no flow or gflow.

    """  # noqa: DOC501
    internal_pattern = pattern.copy()
    og = internal_pattern.extract_opengraph()
    vout = internal_pattern.output_nodes
    measurements: dict[int, mp.Ment] = {}
    meas_planes = internal_pattern.get_meas_plane().items()
    meas_angles = internal_pattern.get_angles()
    for i, plane in meas_planes:
        input_angle = meas_angles[i]
        angle = None if isinstance(input_angle, Expression) else input_angle
        plane_str = str(plane).split(".")[1]  # convert to 'XY', 'YZ', or 'XZ' strings
        if i in vout:
            msg = "Output nodes cannot be measured."
            raise ValueError(msg)
        measurements[i] = mp.Ment(angle, plane_str)
    cflow = og.find_causal_flow()
    gflow = og.find_gflow()
    if not cflow and not gflow:
        msg = "No flow or gflow found, cannot convert to MBQCircuit."
        raise ValueError(msg)
    graph_state: mp.GraphState = mp.GraphState(og.graph)  # type: ignore[no-untyped-call]
    return mp.MBQCircuit(graph_state, input_nodes=list(og.input_nodes)
                         , output_nodes=list(og.output_nodes), measurements=measurements)


def mentpy_to_graphix_pattern(graph_state: mp.MBQCircuit) -> Pattern:  # noqa: D417
    """Convert a MentPy MBQCircuit to a Graphix pattern.

    Parameters
    ----------
    graph_state: mentpy.MBQCircuit

    Returns
    -------
    result: graphix.Pattern

    Exceptions
    ---------
    ValueError
        If the pattern is not available in MentPy to calculate the Lie algebra

    """  # noqa: DOC501
    conversion_dict = {"XY": Plane.XY, "YZ": Plane.YZ, "XZ": Plane.XZ, "X": Plane.XZ, "Y": Plane.YZ, "Z": Plane.XZ}
    measurements: dict[int, Measurement] = {}
    variable_counter = 0
    for index, measurement in graph_state.measurements.items():
        if measurement is None:
            continue
        if measurement.angle is None:
            angle = Placeholder(str("angle_" + str(variable_counter)))
            variable_counter += 1
        elif measurement.plane not in conversion_dict:
            msg = f"Measurement plane {measurement.plane} not supported."
            raise ValueError(msg)
        else:
            angle = float(measurement.angle)
        measurements[index] = Measurement(angle, conversion_dict[measurement.plane])
    open_graph = OpenGraph(graph=graph_state.graph, measurements=measurements
                           , input_nodes=graph_state.input_nodes, output_nodes=graph_state.output_nodes)
    pattern = open_graph.to_pattern()
    pattern.standardize()
    return pattern


def _mentpy_pauli_to_graphix_pauli(generators: list[mp.operators.pauliop.PauliOp]) -> list[list[Pauli]]:
    """Convert a list of MentPy Pauli operators into Graphix format.

    Parameters
    ----------
    generators: list[mentpy.operators.pauliop.PauliOp]
        List of MentPy Pauli operators

    Returns
    -------
    result: list[list[Pauli]]
        List of list of Graphix Pauli operators

    Raises
    ------
    ValueError
        If the element is not a Pauli

    """
    output_generator_list = []
    for generator in generators:
        output_generator = []
        generator_as_list = list(str(generator))
        for pauli in generator_as_list:
            if str(pauli) == "X":
                output_generator.append(Pauli.X)
            elif str(pauli) == "Y":
                output_generator.append(Pauli.Y)
            elif str(pauli) == "Z":
                output_generator.append(Pauli.Z)
            elif str(pauli) == "I":
                output_generator.append(Pauli.I)
            else:
                msg = "The element is not a Pauli"
                raise ValueError(msg)
        output_generator_list.append(output_generator)
    return output_generator_list


def get_lie_algebra(pattern: Pattern) -> list[list[Pauli]]:
    r"""Calculate the Lie algebra for a Graphix MBQC pattern using MentPy utils.

    It is assumed that all parameterised angles in the Graphix pattern are independently tunable.
    This is true even if they are represented by the same parameter :math: `\\theta`.

    Parameters
    ----------
    pattern: Pattern
        Pattern from Graphix

    Returns
    -------
    result: list[list[Pauli]]
        List of list of Graphix Pauli gates

    """
    mp_pattern = graphix_pattern_to_mentpy(pattern)
    if not mp_pattern.trainable_nodes:
        return []
    lie_algebra = mp.utils.calculate_lie_algebra(mp_pattern)
    return _mentpy_pauli_to_graphix_pauli(lie_algebra)  # pyright: ignore[reportArgumentType]


def regenerate_pattern_from_open_graph(pattern: Pattern) -> Pattern:
    """Test function to regenerate pattern from Open Graph through flow-finding algorithm.

    Parameters
    ----------
    pattern: Pattern
        Pattern from Graphix

    Returns
    -------
    result: Pattern
        Pattern from Graphix, calculated from the measurements and underlying Open Graph of the original pattern.

    """
    og_from_pattern = pattern.extract_opengraph()
    return og_from_pattern.to_pattern()
