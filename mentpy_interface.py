"""MBQC pattern according to Measurement Calculus.

ref: MentPy: A Python package for parametrized MBQC circuits
Mantilla Calderón, Luis
https://github.com/mentpy/mentpy
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from graphix import Circuit, Statevec, Pattern
from graphix.ops import Ops
from graphix.pauli import Pauli
from graphix.states import BasicStates
import graphix.instruction as instruction

import mentpy as mp

def mentpy_mbqcircuit_from_graphix_pattern(pattern: Pattern) -> mp.MBQCircuit:
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
        plane_str = str(plane[1]).split('.')[1]  # convert to 'XY', 'YZ', or 'XZ' strings
        if i in vout:
            continue
        measurements[i] = mp.Ment(meas_angles[i], plane_str)
    return mp.MBQCircuit(g, input_nodes=vin, output_nodes=vout, measurements=measurements)


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


def mentpy_pauli_to_graphix_pauli(generators: list[mp.operators.pauliop.PauliOp]) -> list[list[Pauli]]:
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
                raise ValueError("The element is not a Pauli")
        output_generator_list.append(output_generator)
    return output_generator_list


def calculate_lie_algebra(pattern: Pattern):
    lie_algebra = mp.utils.calculate_lie_algebra(mentpy_mbqcircuit_from_graphix_pattern(pattern))
    print(lie_algebra)
    return mentpy_pauli_to_graphix_pauli(lie_algebra)