"""Graphix interface for the MentPy package.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).

ref: MentPy: A Python package for parametrized MBQC circuits
Mantilla Calderón, Luis
https://github.com/mentpy/mentpy
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
from graphix.fundamentals import Plane
from graphix.gflow import find_flow, find_gflow
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.parameter import Expression, ExpressionOrFloat
from graphix.pauli import Pauli

import mentpy as mp

if TYPE_CHECKING:
    from graphix import Pattern


def graphix_pattern_to_mentpy(pattern: Pattern) -> mp.MBQCircuit:
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
    nodes, edges = pattern.get_graph()
    g: nx.Graph[None] = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    vin = pattern.input_nodes if pattern.input_nodes is not None else []
    vout = pattern.output_nodes
    measurements: dict[int, mp.Ment] = {}
    meas_planes = pattern.get_meas_plane().items()
    meas_angles = pattern.get_angles()
    if any(isinstance(angle, Expression) for angle in meas_angles):
        msg = "MentPy doesn't support Expression measurements."
        raise NotImplementedError(msg)
    for i, plane in enumerate(meas_planes):
        plane_str = str(plane[1]).split(".")[1]  # convert to 'XY', 'YZ', or 'XZ' strings
        if i in vout:
            continue
        angle = float(meas_angles[i]) # type: ignore
        measurements[i] = mp.Ment(angle, plane_str) # type: ignore
    flow = find_flow(g, set(vin), set(vout), meas_planes=pattern.get_meas_plane())[0]
    g_flow = find_gflow(g, set(vin), set(vout), meas_planes=pattern.get_meas_plane())[0]
    if not (flow or g_flow):
        msg = "No flow or gflow found, cannot convert to MBQCircuit."
        raise ValueError(msg)
    graph_state: mp.GraphState = mp.GraphState(g)  # type: ignore
    return mp.MBQCircuit(graph_state, input_nodes=vin, output_nodes=vout, measurements=measurements)


def mentpy_to_graphix_pattern(graph_state: mp.MBQCircuit) -> Pattern:
    """Convert a MentPy MBQCircuit to a Graphix pattern.

    Parameters
    ----------
    graph: mentpy.MBQCircuit

    Returns
    -------
    result: graphix.Pattern

    """
    conversion_dict = {"XY": Plane.XY, "YZ": Plane.YZ, "XZ": Plane.XZ}
    measurements: dict[int, Measurement] = {}
    for index, measurement in graph_state.measurements.items():
        if measurement is not None:
            angle = measurement.angle if isinstance(measurement.angle, ExpressionOrFloat) else 0.0
            measurements[index] = Measurement(angle, conversion_dict[measurement.plane])
    open_graph = OpenGraph(graph_state.graph, measurements, graph_state.input_nodes, graph_state.output_nodes)
    return open_graph.to_pattern()


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


def calculate_lie_algebra(pattern: Pattern) -> list[list[Pauli]]:
    """Calculate the Lie algebra for a Graphix MBQC pattern using MentPy utils.

    Parameters
    ----------
    pattern: Pattern
        Pattern from Graphix

    Returns
    -------
    result: list[list[Pauli]]
        List of list of Graphix Pauli gates

    Raises
    ------
    ValueError
        If the pattern is not available in MentPy to calculate the Lie algebra

    """
    mp_pattern = graphix_pattern_to_mentpy(pattern)
    if mp_pattern.trainable_nodes is None:
        msg = "The pattern is not trainable."
        raise ValueError(msg)
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
    og_from_pattern = OpenGraph.from_pattern(pattern)
    return og_from_pattern.to_pattern()
