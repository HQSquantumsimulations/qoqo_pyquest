"""Test qoqo PyQuEST interface for measurements"""
# Copyright © 2019-2021 HQS Quantum Simulations GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
import pytest
import sys
import numpy as np
import numpy.testing as npt
from qoqo import operations as ops
from qoqo import Circuit
from pyquest_cffi import cheat
from pyquest_cffi import utils
from typing import (
    Dict,
    Optional,
    List
)
from copy import copy
from qoqo_pyquest import (
    pyquest_call_operation,
    pyquest_call_circuit
)


def test_measurement():
    """Test gate operations without free parameters with PyQuEST interface"""
    op = ops.MeasureQubit(qubit=0, readout='test', readout_index=0)
    test_dict = {'test': np.array([0, 0, 0, 0])}
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(2, env)
    state = np.array([0, 1, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(op, qubits,
                           classical_bit_registers=test_dict,
                           classical_float_registers=dict(),
                           classical_complex_registers=dict(),
                           output_bit_register_dict=dict(),)
    utils.destroyQureg()(qubits, env=env)
    utils.destroyQuestEnv()(env)
    assert test_dict['test'][0]


def test_pragma_repeated_measurement_pyquest() -> None:
    """Test repeated measurement with PyQuEST interface"""
    op = ops.PragmaRepeatedMeasurement
    operation = op(readout='ro',
                   qubit_mapping={0: 0},
                   number_measurements=100)
    test_dict: Dict[str, List[List[bool]]] = {'ro': list(list())}
    test_dict2: Dict[str, List[bool]] = {'ro': list()}
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(2, env)
    state = 1 / np.sqrt(2) * np.array([1, 1, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(operation=operation, qureg=qubits, classical_bit_registers=test_dict2,
                           classical_float_registers=dict(),
                           classical_complex_registers=dict(),
                           output_bit_register_dict=test_dict,)
    utils.destroyQureg()(qubits, env=env)
    utils.destroyQuestEnv()(env)
    assert len(test_dict['ro']) == 100


def test_pragma_get_pauli_product_pyquest() -> None:
    """Test repeated measurement with PyQuEST interface"""

    operation = ops.PragmaGetPauliProduct(readout='ro',
                                          qubit_paulis={0: 3},
                                          circuit=Circuit()
                                          )
    test_dict: Dict[str, List[float]] = {'ro': [0.0]}
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(2, env)
    state = np.array([1, 0, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(operation=operation, qureg=qubits, classical_bit_registers=dict(),
                           classical_float_registers=test_dict,
                           classical_complex_registers=dict(),
                           output_bit_register_dict=dict(),)
    assert test_dict['ro'][0] == 1

    test_dict = {'ro': list()}
    state = np.array([0, 1, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(operation=operation, qureg=qubits, classical_bit_registers=dict(),
                           classical_float_registers=test_dict,
                           classical_complex_registers=dict(),
                           output_bit_register_dict=dict(),)

    assert test_dict['ro'][0] == -1

    test_dict = {'ro': list()}
    state = 1 / np.sqrt(2) * np.array([1, 1, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(operation=operation, qureg=qubits, classical_bit_registers=dict(),
                           classical_float_registers=test_dict,
                           classical_complex_registers=dict(),
                           output_bit_register_dict=dict(),)
    assert test_dict['ro'][0] == 0

    operation = ops.PragmaGetPauliProduct(readout='ro',
                                          qubit_paulis={0: 3, 1: 3},
                                          circuit=Circuit()
                                          )
    test_dict = {'ro': list()}
    state = 1 / np.sqrt(2) * np.array([0, 1, 1, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(operation=operation, qureg=qubits, classical_bit_registers=dict(),
                           classical_float_registers=test_dict,
                           classical_complex_registers=dict(),
                           output_bit_register_dict=dict(),)
    assert np.isclose(test_dict['ro'][0], -1)

    test_dict = {'ro': list()}
    state = 1 / np.sqrt(2) * np.array([0, 1, -1, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(operation=operation, qureg=qubits, classical_bit_registers=dict(),
                           classical_float_registers=test_dict,
                           classical_complex_registers=dict(),
                           output_bit_register_dict=dict(),)
    assert np.isclose(test_dict['ro'][0], -1)

    utils.destroyQureg()(qubits=qubits, env=env)
    utils.destroyQuestEnv()(env)


@pytest.mark.parametrize("init", [
    (None, 1 / np.sqrt(2) * np.array([1, 1, 0, 0]), ''),
])
def test_pragma_get_statevec_pyquest(init) -> None:
    """Test get state vector with PyQuEST interface"""
    op = ops.PragmaGetStateVector
    test_dict: Dict[str, List[complex]] = {'ro': [0, 0, 0, 0]}
    operation = op(readout='ro',
                   circuit=Circuit(),
                   )
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(2, env)
    state = 1 / np.sqrt(2) * np.array([1, 1, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))

    pyquest_call_operation(operation=operation, qureg=qubits, classical_bit_registers=dict(),
                           classical_float_registers=dict(),
                           classical_complex_registers=test_dict,
                           output_bit_register_dict=dict(),)
    utils.destroyQureg()(qubits=qubits, env=env)
    utils.destroyQuestEnv()(env)
    npt.assert_array_almost_equal(test_dict['ro'], init[1], decimal=4)


def test_pragma_get_densitymatrix_pyquest() -> None:
    """Test get density matrix with PyQuEST interface"""
    op = ops.PragmaGetDensityMatrix
    test_dict: Dict[str, List[complex]] = {'ro': [0, 0, 0, 0]}
    operation = op(readout='ro',
                   circuit=Circuit()
                   )
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(1, env)
    state = 1 / np.sqrt(2) * np.array([1, 1])
    density_matrix = 1 / 2 * np.array([[1, 1], [1, 1]]).flatten()
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))

    pyquest_call_operation(operation=operation, qureg=qubits, classical_bit_registers=dict(),
                           classical_float_registers=dict(),
                           classical_complex_registers=test_dict,
                           output_bit_register_dict=dict(),)
    utils.destroyQureg()(qubits=qubits, env=env)
    utils.destroyQuestEnv()(env)
    npt.assert_array_almost_equal(test_dict['ro'], density_matrix)


@pytest.mark.parametrize("init", [(None, np.array([0, 1, 0, 0]), ''),
                                  ])
def test_pragma_get_occupation_probability_pyquest(init) -> None:
    """Test get occupation probability with PyQuEST interface"""
    op = ops.PragmaGetOccupationProbability
    test_dict: Dict[str, List[float]] = {'ro': [0, 0]}
    operation = op(readout='ro',
                    circuit=Circuit()
                   )
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(2, env)
    state = np.array([0, 1, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))

    pyquest_call_operation(operation=operation, qureg=qubits, classical_bit_registers=dict(),
                           classical_float_registers=test_dict,
                           classical_complex_registers=dict(),
                           output_bit_register_dict=dict(),)
    utils.destroyQureg()(qubits=qubits, env=env)
    utils.destroyQuestEnv()(env)
    npt.assert_array_almost_equal(test_dict['ro'], init[1])




@pytest.mark.parametrize("init", [(np.array([1, 0, 0, 0]), 1, {0: 3}),
                                  (np.array([0, 0, 1, 0]), -1, {1: 3}),
                                  (np.array([0, 1, 0, 0]), -1, {0: 3}),
                                  (np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0]), 0, {0: 3}),
                                  (np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
                                   1, {0: 3, 1: 3}),
                                  (np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0]),
                                   - 1, {0: 3, 1: 3}),
                                  ])
def test_pragma_get_pauli_prod_measurement_pyquest(init) -> None:
    """Test measuring product of pauli operators with PyQuEST interface"""
    test_dict: Dict[str, List[float]] = {'ro': [0, 0, 0, 0]}

    op = ops.PragmaGetPauliProduct

    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(2, env)
    state = init[0]
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    operation = op(readout='ro',
                   qubit_paulis=init[2],
                   circuit=Circuit())

    pyquest_call_operation(operation=operation, qureg=qubits, classical_bit_registers=dict(),
                           classical_float_registers=test_dict,
                           classical_complex_registers=dict(),
                           output_bit_register_dict=dict(),)
    npt.assert_array_almost_equal(test_dict['ro'], init[1])
    utils.destroyQureg()(qubits, env=env)
    utils.destroyQuestEnv()(env)


if __name__ == '__main__':
    pytest.main(sys.argv)
