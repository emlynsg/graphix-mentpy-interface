"""Graphix interface for the MentPy package.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).

ref: MentPy: A Python package for parametrized MBQC circuits
Mantilla Calderón, Luis
https://github.com/mentpy/mentpy
"""
from __future__ import annotations

import networkx as nx
from graphix import Pattern, command
from graphix.fundamentals import Plane
from graphix.gflow import find_flow
from graphix.opengraph import OpenGraph
from graphix.pauli import Pauli

import mentpy as mp


def graphix_pattern_to_mentpy(pattern: Pattern) -> mp.MBQCircuit:
    """Convert a Graphix pattern to a MentPy MBQCircuit.

    Parameters
    ----------
    pattern: graphix.Pattern
        Graphix pattern object

    Returns
    -------
    result: mentpy.MBQCircuit
        MentPy MBQCircuit object

    """
    nodes, edges = pattern.get_graph()
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    vin = pattern.input_nodes if pattern.input_nodes is not None else []
    vout = pattern.output_nodes
    measurements = {}
    meas_planes = pattern.get_meas_plane().items()
    meas_angles = pattern.get_angles()
    for i, plane in enumerate(meas_planes):
        plane_str = str(plane[1]).split(".")[1]  # convert to 'XY', 'YZ', or 'XZ' strings
        if i in vout:
            continue
        measurements[i] = mp.Ment(meas_angles[i], plane_str)
    return mp.MBQCircuit(g, input_nodes=vin, output_nodes=vout, measurements=measurements)


def mentpy_to_graphix_pattern(graph: mp.MBQCircuit) -> Pattern:
    """Convert a MentPy MBQCircuit to a Graphix pattern.

    Parameters
    ----------
    graph: mentpy.MBQCircuit
    output_generator_list = []

    Returns
    -------
    result: graphix.Pattern
        Graphix pattern object

    """
    conversion_dict = {"XY": Plane.XY, "YZ": Plane.YZ, "XZ": Plane.XZ}
    new_nodes: list[int] = list(graph.inputc)
    edges: list[tuple[int, int]] = list(graph.edges)
    input_nodes = graph.input_nodes
    m_commands: dict[int, command.M] = {}
    if graph.measurements is not None:
        for node, measurement in graph.measurements.items():
            if measurement is not None:
                m_commands[node] = command.M(node, plane=conversion_dict[measurement.plane], angle=measurement.angle)
    for node in graph.outputc:
        if node not in graph.measurements:
            m_commands[node] = command.M(node)
    pattern = Pattern(input_nodes, None, None)
    pattern.extend((command.N(node) for node in new_nodes), (command.E(edge) for edge in edges), (m_command for m_command in m_commands.values()))
    return pattern

# def mentpy_pauli_to_graphix_ops(generators: list[mp.operators.pauliop.PauliOp]) -> list[list[Ops]]:
#     output_generator_list = []
#     for generator in generators:
#         output_generator = []
#         generator_as_list = list(str(generator))
#         for pauli in generator_as_list:
#             if str(pauli) == "X":
#                 output_generator.append(Ops.X)
#             elif str(pauli) == "Y":
#                 output_generator.append(Ops.Y)
#             elif str(pauli) == "Z":
#                 output_generator.append(Ops.Z)
#             elif str(pauli) == "I":
#                 output_generator.append(Ops.I)
#             else:
#                 raise ValueError("The element is not a Pauli")
#         output_generator_list.append(output_generator)
#     return output_generator_list


def _mentpy_pauli_to_graphix_pauli(generators: list[mp.operators.pauliop.PauliOp]) -> list[list[Pauli]]:
    """Helper function that converts a list of MentPy Pauli operators into Graphix format.
    
    Parameters
    ----------
    generators: list[mentpy.operators.pauliop.PauliOp]
        List of MentPy Pauli operators
    
    Returns
    -------
    result: list[list[Pauli]]
        List of list of Graphix Pauli operators

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

    """
    og = OpenGraph.from_pattern(pattern)
    f, _layers = find_flow(
        og.inside, set(og.inputs), set(og.outputs), {node: meas.plane for node, meas in og.measurements.items()}
    )
    if f is None:
        raise ValueError("Open Graph doesn't have flow. Flow required to generate Lie algebra.")
    lie_algebra = mp.utils.calculate_lie_algebra(graphix_pattern_to_mentpy(pattern))
    return _mentpy_pauli_to_graphix_pauli(lie_algebra)
