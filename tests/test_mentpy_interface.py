"""Tests for transpiler from Graphix-MentPy interface.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

import logging
from math import pi

import numpy as np
import pytest
from graphix import Pattern, instruction
from graphix.opengraph import OpenGraph
from graphix.random_objects import rand_circuit
from graphix.transpiler import Circuit
from numpy.random import PCG64, Generator

from graphix_mentpy_interface import calculate_lie_algebra, graphix_pattern_to_mentpy, mentpy_to_graphix_pattern, regenerate_pattern_from_open_graph

logger = logging.getLogger(__name__)

TEST_BASIC_CIRCUITS = [
    Circuit(2, instr=[instruction.H(0), instruction.CNOT(0, 1)]),
    Circuit(2, instr=[instruction.S(0), instruction.CNOT(0, 1)]),
    Circuit(2, instr=[instruction.X(0), instruction.CNOT(0, 1)]),
    Circuit(2, instr=[instruction.Y(0), instruction.CNOT(0, 1)]),
    Circuit(2, instr=[instruction.Z(0), instruction.CNOT(0, 1)]),
    Circuit(2, instr=[instruction.RX(0, pi / 4), instruction.CNOT(0, 1)]),
    Circuit(2, instr=[instruction.RY(0, pi / 4), instruction.CNOT(0, 1)]),
    Circuit(2, instr=[instruction.RZ(0, pi / 4), instruction.CNOT(0, 1)]),
    Circuit(2, instr=[instruction.CNOT(0, 1), instruction.CNOT(0, 1)]),
    Circuit(3, instr=[instruction.CCX(0, (1, 2))]),
]

def regenerate_pattern_from_og(pattern: Pattern) -> Pattern:
    """Put pattern through regeneration via og."""
    og_from_pattern = OpenGraph.from_pattern(pattern)
    return og_from_pattern.to_pattern()


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_check(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation comparing state vector back-end, where a copy has been sent to MentPy and back."""
    pattern = circuit.transpile().pattern
    pattern_og = regenerate_pattern_from_open_graph(pattern)
    pattern.minimize_space()
    pattern_og.minimize_space()
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    state_mbqc_og = pattern_og.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state_mbqc_og.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation comparing state vector back-end, where a copy has been sent to MentPy and back."""
    pattern = circuit.transpile().pattern
    pattern_to_and_from_mentpy = mentpy_to_graphix_pattern(graphix_pattern_to_mentpy(pattern))
    pattern_to_and_from_mentpy.standardize()
    pattern_to_and_from_mentpy.shift_signals()
    pattern_to_and_from_mentpy_mbqc = pattern_to_and_from_mentpy.simulate_pattern(rng=fx_rng)
    pattern = regenerate_pattern_from_og(pattern)
    pattern.minimize_space()
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), pattern_to_and_from_mentpy_mbqc.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_lie_algebra(circuit: Circuit) -> None:
    """Test transpiled and reconverted circuits have flow and that Lie algebra generation works."""
    pattern = circuit.transpile().pattern
    algebra = calculate_lie_algebra(pattern)
    assert algebra is not None


@pytest.mark.parametrize("jumps", range(1, 6))
@pytest.mark.parametrize("check", ["simulation", "lie_algebra"])
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
