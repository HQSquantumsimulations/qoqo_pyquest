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
from qoqo.operations import OperationNotInBackendError
from typing import (
    Dict,
    Optional
)
from hqsbase.calculator import Calculator
from copy import copy
from qoqo_pyquest import (
    pyquest_call_operation,
    pyquest_call_circuit
)
from qoqo.registers import (
    BitRegister,
    FloatRegister,
    ComplexRegister,
)


def test_measurement():
    """Test gate operations without free parameters with PyQuEST interface"""
    op = ops.MeasureQubit(qubit=0, readout='test', readout_index=0)
    definition = ops.Definition('test', 'bit', length=2,)
    test_dict = {'test': BitRegister(definition)}
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(2, env)
    state = np.array([0, 1, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(op, qubits, test_dict)
    utils.destroyQureg()(qubits, env=env)
    utils.destroyQuestEnv()(env)
    assert test_dict['test'].register[0]


def test_pragma_repeated_measurement_pyquest() -> None:
    """Test repeated measurement with PyQuEST interface"""
    op = ops.PragmaRepeatedMeasurement
    operation = op(readout='ro',
                   qubit_mapping={0: 0},
                   number_measurements=100)
    definition = ops.Definition('ro', 'bit', length=2,)
    test_dict = {'ro': BitRegister(definition)}
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(2, env)
    state = 1 / np.sqrt(2) * np.array([1, 1, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(operation=operation, qureg=qubits, classical_registers=test_dict)
    utils.destroyQureg()(qubits, env=env)
    utils.destroyQuestEnv()(env)
    assert test_dict['ro'].register.shape == (100, 1)

def test_pragma_pauli_product_measurement_pyquest() -> None:
    """Test repeated measurement with PyQuEST interface"""
    op = ops.PragmaGetPauliProduct

    operation = op(readout='ro',
                   pauli_product=[1, 0],
                   )
    definition = ops.Definition('ro', 'float', length=1,)
    test_dict = {'ro': FloatRegister(definition)}
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(2, env)
    state = np.array([1, 0, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(operation=operation, qureg=qubits, classical_registers=test_dict)
    assert test_dict['ro'].register[0] == 1

    test_dict = {'ro': FloatRegister(definition)}
    state = np.array([0, 1, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(operation=operation, qureg=qubits, classical_registers=test_dict)
    assert np.isclose(test_dict['ro'].register[0], -1)

    test_dict = {'ro': FloatRegister(definition)}
    state = 1 / np.sqrt(2) * np.array([1, 1, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(operation=operation, qureg=qubits, classical_registers=test_dict)
    assert test_dict['ro'].register[0] == 0

    operation = op(readout='ro',
                   pauli_product=[0, 1],
                   )
    test_dict = {'ro': FloatRegister(definition)}
    state = 1 / np.sqrt(2) * np.array([0, 1, 1, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(operation=operation, qureg=qubits, classical_registers=test_dict)
    assert np.isclose(test_dict['ro'].register[0], -1)

    test_dict = {'ro': FloatRegister(definition)}
    state = 1 / np.sqrt(2) * np.array([0, 1, -1, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    pyquest_call_operation(operation=operation, qureg=qubits, classical_registers=test_dict)
    assert np.isclose(test_dict['ro'].register[0], -1)

    utils.destroyQureg()(qubits=qubits, env=env)
    utils.destroyQuestEnv()(env)


@pytest.mark.parametrize("init", [
    (None, 1 / np.sqrt(2) * np.array([1, 1, 0, 0]), ''),
    ({0: 0, 1: 1}, 1 / np.sqrt(2) * np.array([1, 1, 0, 0]), '(0:0,1:1,)'),
    ({0: 1, 1: 0}, 1 / np.sqrt(2) * np.array([1, 0, 1, 0]), '(0:1,1:0,)'),
])
def test_pragma_get_statevec_pyquest(init) -> None:
    """Test get state vector with PyQuEST interface"""
    op = ops.PragmaGetStateVector
    definition = ops.Definition('ro', 'complex', length=4,)
    test_dict = {'ro': ComplexRegister(definition)}
    operation = op(readout='ro',
                   qubit_mapping=init[0]
                   )
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(2, env)
    state = 1 / np.sqrt(2) * np.array([1, 1, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))

    pyquest_call_operation(operation=operation, qureg=qubits, classical_registers=test_dict)
    utils.destroyQureg()(qubits=qubits, env=env)
    utils.destroyQuestEnv()(env)
    npt.assert_array_almost_equal(test_dict['ro'].register, init[1], decimal=4)

def test_pragma_get_densitymatrix_pyquest() -> None:
    """Test get density matrix with PyQuEST interface"""
    op = ops.PragmaGetDensityMatrix
    definition = ops.Definition('ro', 'complex', length=4,)
    test_dict = {'ro': ComplexRegister(definition)}
    operation = op(readout='ro',
                   )
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(1, env)
    state = 1 / np.sqrt(2) * np.array([1, 1])
    density_matrix = 1 / 2 * np.array([[1, 1], [1, 1]])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))

    pyquest_call_operation(operation=operation, qureg=qubits, classical_registers=test_dict)
    utils.destroyQureg()(qubits=qubits, env=env)
    utils.destroyQuestEnv()(env)
    npt.assert_array_almost_equal(test_dict['ro'].register, density_matrix)

@pytest.mark.parametrize("init", [(None, np.array([0, 1, 0, 0]), ''),
                                  ({0: 0, 1: 1}, np.array([0, 1, 0, 0]), '(0:0,1:1,)'),
                                  ({0: 1, 1: 0}, np.array([0, 0, 1, 0]), '(0:1,1:0,)'),
                                  ])
def test_pragma_get_occupation_probability_pyquest(init) -> None:
    """Test get occupation probability with PyQuEST interface"""
    op = ops.PragmaGetOccupationProbability
    definition = ops.Definition('ro', 'float', length=4,)
    test_dict = {'ro': FloatRegister(definition)}
    operation = op(readout='ro',
                   qubit_mapping=init[0]
                   )
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(2, env)
    state = np.array([0, 1, 0, 0])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))

    pyquest_call_operation(operation=operation, qureg=qubits, classical_registers=test_dict)
    utils.destroyQureg()(qubits=qubits, env=env)
    utils.destroyQuestEnv()(env)
    npt.assert_array_almost_equal(test_dict['ro'].register, init[1])

@pytest.mark.parametrize("init", [np.array([1, 0]),
                                  ])
def test_pragma_get_rotated_occupation_probability_pyquest(init) -> None:
    """Test get rotated occupation probability with PyQuEST interface"""
    op = ops.PragmaGetRotatedOccupationProbability
    definition = ops.Definition('ro', 'float', length=4,)
    test_dict = {'ro': FloatRegister(definition)}
    circuit = Circuit()
    circuit += ops.Hadamard(qubit=0)
    operation = op(readout='ro',
                   circuit=circuit
                   )
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(1, env)
    state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))

    pyquest_call_operation(operation=operation, qureg=qubits, classical_registers=test_dict)
    npt.assert_array_almost_equal(test_dict['ro'].register, init)
    utils.destroyQureg()(qubits, env=env)
    utils.destroyQuestEnv()(env)

def test_pragma_pauli_prod_measurement_pyquest() -> None:
    """Test measuring product of pauli operators with PyQuEST interface"""
    definition = ops.Definition('ro', 'float', length=1,)
    test_dict = {'ro': FloatRegister(definition)}
    op = ops.PragmaPauliProdMeasurement
    operation = op(readout='ro',
                   readout_index=0,
                   qubits=[0],
                   paulis=[1]
                   )
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(1, env)
    state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))

    pyquest_call_operation(operation=operation, qureg=qubits, classical_registers=test_dict)
    npt.assert_array_almost_equal(test_dict['ro'].register, 1)
    utils.destroyQureg()(qubits, env=env)
    utils.destroyQuestEnv()(env)

@pytest.mark.parametrize("init", [(np.array([1, 0, 0, 0]), 1, [1, 0]),
                                  (np.array([0, 0, 1, 0]), -1, [0, 1]),
                                  (np.array([0, 1, 0, 0]), -1, [0, 1]),
                                  (np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0]), 0, [0, 1]),
                                  (np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]), 1, [0, 1]),
                                  (np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0]), -1, [0, 1]),
                                  ])
def test_pragma_get_pauli_prod_measurement_pyquest(init) -> None:
    """Test measuring product of pauli operators with PyQuEST interface"""
    name1 = 'PragmaGetPauliProduct ro'
    definition = ops.Definition('ro', 'float', length=1,)
    test_dict = {'ro': FloatRegister(definition)}

    op = ops.PragmaGetPauliProduct
    operation = op(readout='ro',
                   pauli_product=[0, 1]
                   )

    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(2, env)
    state = init[0]
    cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
    operation = op(readout='ro',
                   pauli_product=init[2]
                   )

    pyquest_call_operation(operation=operation, qureg=qubits, classical_registers=test_dict)
    npt.assert_array_almost_equal(test_dict['ro'].register, init[1])
    utils.destroyQureg()(qubits, env=env)
    utils.destroyQuestEnv()(env)


if __name__ == '__main__':
    pytest.main(sys.argv)
