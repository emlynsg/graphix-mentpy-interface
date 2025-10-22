"""Tests for transpiler from Graphix-MentPy interface.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

import logging
from math import pi

import numpy as np
import pytest
from graphix import instruction, transpiler
from graphix.fundamentals import Plane
from graphix.gflow import find_flow
from graphix.opengraph import OpenGraph
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod
from graphix.transpiler import Circuit
from numpy.random import PCG64, Generator

from graphix_mentpy_interface import calculate_lie_algebra, graphix_pattern_to_mentpy, mentpy_to_graphix_pattern

logger = logging.getLogger(__name__)

TEST_BASIC_CIRCUITS = [
    Circuit(1, instr=[instruction.H(0)]),
    Circuit(1, instr=[instruction.S(0)]),
    Circuit(1, instr=[instruction.X(0)]),
    Circuit(1, instr=[instruction.Y(0)]),
    Circuit(1, instr=[instruction.Z(0)]),
    Circuit(1, instr=[instruction.I(0)]),
    Circuit(1, instr=[instruction.RX(0, pi / 4)]),
    Circuit(1, instr=[instruction.RY(0, pi / 4)]),
    Circuit(1, instr=[instruction.RZ(0, pi / 4)]),
    Circuit(2, instr=[instruction.CNOT(0, 1)]),
    Circuit(3, instr=[instruction.CCX(0, (1, 2))]),
    Circuit(2, instr=[instruction.RZZ(0, 1, pi / 4)]),
]


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation comparing state vector back-end, where a copy has been sent to MentPy and back."""
    pattern = circuit.transpile().pattern
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    pattern_to_and_from_mentpy = mentpy_to_graphix_pattern(graphix_pattern_to_mentpy(pattern))
    pattern_to_and_from_mentpy_mbqc = pattern_to_and_from_mentpy.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), pattern_to_and_from_mentpy_mbqc.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation_statevec(circuit: Circuit) -> None:
    """Test circuit transpilation comparing state vector back-end, where a copy has been sent to MentPy and back."""
    pattern = circuit.transpile().pattern
    state = circuit.simulate_statevector().statevec
    state_to_and_from_mentpy = mentpy_to_graphix_pattern(graphix_pattern_to_mentpy(pattern)).simulate_statevector().statevec
    assert np.abs(np.dot(state.flatten().conjugate(), state_to_and_from_mentpy.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_lie_algebra(circuit: Circuit) -> None:
    """Test transpiled and reconverted circuits have flow and that Lie algebra generation works."""
    pattern = circuit.transpile().pattern
    pattern_to_and_from_mentpy = mentpy_to_graphix_pattern(graphix_pattern_to_mentpy(pattern))
    og = OpenGraph.from_pattern(pattern_to_and_from_mentpy)
    f, _layers = find_flow(
        og.inside, set(og.inputs), set(og.outputs), {node: meas.plane for node, meas in og.measurements.items()}
    )
    assert f is not None
    algebra = calculate_lie_algebra(pattern)
    assert algebra is not None
    assert not any(algebra)


@pytest.mark.parametrize("jumps", range(1, 6))
@pytest.mark.parametrize("check", ["simulation", "flow", "lie_algebra"])
def test_random_circuit(fx_bg: PCG64, jumps: int, check: str) -> None:
    """Test random circuit transpilation and conversion."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 6
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
    if check == "simulation":
        test_circuit_simulation(circuit, rng)
    elif check == "lie_algebra":
        test_circuit_lie_algebra(circuit)
