"""Tests for transpiler from Graphix-MentPy interface.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

import logging
from math import pi

import numpy as np
import pytest
from graphix import Pattern, instruction
from graphix.command import E, M, N
from graphix.parameter import Placeholder
from graphix.random_objects import rand_circuit
from graphix.transpiler import Circuit
from numpy.random import PCG64, Generator

from graphix_mentpy_interface import (
    get_lie_algebra,
    graphix_pattern_to_mentpy,
    mentpy_to_graphix_pattern,
    regenerate_pattern_from_open_graph,
)

logger = logging.getLogger(__name__)

TEST_BASIC_CIRCUITS = [
    Circuit(2, instr=[instruction.H(0)]),
    Circuit(2, instr=[instruction.S(0)]),
    Circuit(2, instr=[instruction.X(0)]),
    Circuit(2, instr=[instruction.Y(0)]),
    Circuit(2, instr=[instruction.Z(0)]),
    Circuit(2, instr=[instruction.RX(0, pi / 4)]),
    Circuit(2, instr=[instruction.RY(0, pi / 4)]),
    Circuit(2, instr=[instruction.RZ(0, pi / 4)]),
    Circuit(2, instr=[instruction.CNOT(0, 1)]),
    Circuit(3, instr=[instruction.CCX(0, (1, 2))]),
]

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


def test_circuit_parameters() -> None:
    """Tests if a parameterized circuit retains its parameters after transpilation and conversion to and from MentPy."""
    alpha = Placeholder("alpha")
    beta = Placeholder("beta")
    gamma = Placeholder("gamma")
    circuit = Circuit(2, instr=[instruction.RX(0, alpha), instruction.RY(1, beta), instruction.RZ(0, gamma)])
    pattern = regenerate_pattern_from_open_graph(circuit.transpile().pattern)
    pattern_to_and_from_mentpy = mentpy_to_graphix_pattern(graphix_pattern_to_mentpy(pattern))
    assert pattern_to_and_from_mentpy.is_parameterized  # type: ignore[truthy-function]


def test_pattern_measurement_parameter() -> None:
    """Tests if a pattern with a measurement parameter retains its parameter after conversion to and from MentPy."""
    pattern = Pattern(input_nodes=[0])
    pattern.add(N(node=1))
    pattern.add(E((0, 1)))
    pattern.add(M(node=0))
    pattern.add(N(node=2))
    pattern.add(E((1, 2)))
    alpha = Placeholder("alpha")
    pattern.add(M(node=1, angle=alpha))
    pattern.add(N(node=3))
    pattern.add(E((2, 3)))
    beta = Placeholder("beta")
    pattern.add(M(node=2, angle=beta))
    pattern_to_and_from_mentpy = mentpy_to_graphix_pattern(graphix_pattern_to_mentpy(pattern))
    assert pattern_to_and_from_mentpy.is_parameterized  # type: ignore[truthy-function]


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation comparing state vector back-end, where a copy has been sent to MentPy and back."""
    pattern = regenerate_pattern_from_open_graph(circuit.transpile().pattern)
    pattern_to_and_from_mentpy = mentpy_to_graphix_pattern(graphix_pattern_to_mentpy(pattern))
    pattern_to_and_from_mentpy.standardize()
    pattern_to_and_from_mentpy_mbqc = pattern_to_and_from_mentpy.simulate_pattern(rng=fx_rng)
    pattern = regenerate_pattern_from_open_graph(pattern)
    pattern.minimize_space()
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), pattern_to_and_from_mentpy_mbqc.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_lie_algebra(circuit: Circuit) -> None:
    """Test transpiled and reconverted circuits have flow and that Lie algebra generation works."""
    pattern = regenerate_pattern_from_open_graph(circuit.transpile().pattern)
    algebra = get_lie_algebra(pattern)
    if not pattern.is_parameterized:  # type: ignore[truthy-function]
        assert not algebra
    else:
        assert algebra is not None


@pytest.mark.parametrize("jumps", range(1, 6))
def test_random_circuit(fx_bg: PCG64, jumps: int) -> None:
    """Test random circuit transpilation and conversion."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 6
    parameters = [Placeholder("theta"), Placeholder("phi")]
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True, parameters=parameters)
    test_circuit_lie_algebra(circuit)
