"""Test qoqo PyQuEST interface"""
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


@pytest.mark.parametrize("init", [
    ops.Hadamard,
    ops.PauliX,
    ops.PauliY,
    ops.PauliZ,
    ops.SGate,
    ops.TGate,
    ops.SqrtPauliX,
    ops.InvSqrtPauliX,
    ops.CNOT,
    ops.SWAP,
    ops.ControlledPauliZ,
    ops.ControlledPauliY,
    ops.MolmerSorensenXX,
    ops.InvSqrtISwap,
    ops.SqrtISwap,
])
def test_simple_gate_matrices(init):
    """Test gate operations without free parameters with PyQuEST interface"""
    op = init
    matrix_gate = op.unitary_matrix_from_parameters()
    if op.number_of_qubits() == 1:
        q0 = 0
        operation = op(qubit=q0)
    else:
        q0, q1 = (0, 1)
        operation = op(control=q1, qubit=q0)

    # Testing gate functionality with pyquest
    if matrix_gate.shape == (2, 2):
        matrix_reconstructed_Xmon = build_one_qubit_matrix_Xmon(operation, {})
        try:
            matrix_reconstructed = build_one_qubit_matrix(operation, {})
        except OperationNotInBackendError:
            matrix_reconstructed = None
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed_Xmon = build_two_qubit_matrix_Xmon(operation, {})
        try:
            matrix_reconstructed = build_two_qubit_matrix(operation, {})
        except OperationNotInBackendError:
            matrix_reconstructed = None
    if matrix_reconstructed is not None:
        npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)
    matrix_to_test = (matrix_gate @ matrix_reconstructed_Xmon.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))


@pytest.mark.parametrize("init", [
    ops.RotateX,
    ops.RotateY,
    ops.RotateZ,
    ops.RotateAroundSphericalAxis,
    ops.W,
    ops.ControlledPhaseShift,
    ops.SingleQubitGate,
])
def test_parameterised_gate_matrices(init):
    """Test gate operations with free parameters with PyQuEST interface"""
    op = init
    matrix_args = {'theta': -np.pi, 'spherical_theta': np.pi, 'spherical_phi': np.pi,
                   'alpha_r': 1, 'alpha_i': 0, 'beta_r': 0, 'beta_i': 0}
    matrix_gate = op.unitary_matrix_from_parameters(**matrix_args)
    if op.number_of_qubits() == 1:
        q0 = 0
        operation = op(qubit=q0, **matrix_args)
    else:
        q0, q1 = (0, 1)
        operation = op(control=q1, qubit=q0, **matrix_args)

    # Testing gate functionality with pyquest
    if matrix_gate.shape == (2, 2):
        matrix_reconstructed = build_one_qubit_matrix(operation, {})
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed = build_two_qubit_matrix(operation, {})
    if matrix_reconstructed is not None:
        npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)
    matrix_to_test = (matrix_gate @ matrix_reconstructed.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))


@pytest.mark.parametrize(
    "init",
    [ops.SingleQubitGate,
     ])
@pytest.mark.parametrize("a", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
@pytest.mark.parametrize("b", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
@pytest.mark.parametrize("c", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
@pytest.mark.parametrize("d", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
def test_single_qubit_gate(init, a, b, c, d) -> None:
    """Test general single qubit gate operation with PyQuEST interface"""
    op = init
    alpha = np.exp(1j * a) * np.cos(b)
    beta = np.exp(1j * c) * np.sin(b)
    matrix_gate = op.unitary_matrix_from_parameters(alpha_r=np.real(
        alpha), alpha_i=np.imag(alpha), beta_r=np.real(beta), beta_i=np.imag(beta))
    Alpha_r, Alpha_i, Beta_r, Beta_i, Global_phase, q0 = (
        ('alpha_r', 'alpha_i', 'beta_r', 'beta_i', 'global_phase', 0))
    operation = op(qubit=q0, alpha_r=Alpha_r, alpha_i=Alpha_i,
                   beta_r=Beta_r, beta_i=Beta_i, global_phase=Global_phase)

    alpha = np.exp(1j * a) * np.cos(b)
    beta = np.exp(1j * c) * np.sin(b)
    substitution_dict = {
        'alpha_r': np.real(alpha),
        'alpha_i': np.imag(alpha),
        'beta_r': np.real(beta),
        'beta_i': np.imag(beta),
        'global_phase': d}

    matrix_reconstructed = build_one_qubit_matrix(operation, {}, substitution_dict)
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)

    operation.substitute_parameters(substitution_dict)

    matrix_reconstructed = build_one_qubit_matrix(operation, {})
    matrix_reconstructed_Xmon = build_one_qubit_matrix_Xmon(operation, {})

    matrix_to_test = (matrix_gate @ matrix_reconstructed_Xmon.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)


@pytest.mark.parametrize("init", [ops.RotateX,
                                  ops.RotateY,
                                  ops.RotateZ,
                                  ops.ControlledPhaseShift
                                  ])
@pytest.mark.parametrize("theta_p", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_single_parameter_gate_matrices(init, theta_p) -> None:
    """Test gate operations with single parameter with PyQuEST interface"""
    op = init
    matrix_gate = op.unitary_matrix_from_parameters(theta=theta_p)
    if op.number_of_qubits() == 1:
        theta, q0 = ('theta', 0)
        operation = op(qubit=q0, theta=theta)
    else:
        (theta, q0, q1) = ('theta', 0, 1)
        operation = op(control=q1, qubit=q0, theta=theta)

    substitution_dict = {'theta': theta_p}
    if matrix_gate.shape == (2, 2):
        matrix_reconstructed = build_one_qubit_matrix(operation, {}, substitution_dict)
    else:
        matrix_reconstructed = build_two_qubit_matrix(operation, {}, substitution_dict)
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)

    operation.substitute_parameters(substitution_dict)

    if matrix_gate.shape == (2, 2):
        matrix_reconstructed = build_one_qubit_matrix(operation, {})
        matrix_reconstructed_Xmon = build_one_qubit_matrix_Xmon(operation, {})

    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed = build_two_qubit_matrix(operation, {})
        matrix_reconstructed_Xmon = build_two_qubit_matrix_Xmon(operation, {})

    matrix_to_test = (matrix_gate @ matrix_reconstructed_Xmon.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)


@pytest.mark.parametrize("init", [ops.W,
                                  ])
@pytest.mark.parametrize("theta_p", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
@pytest.mark.parametrize("phi_p", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_parameter_gate_matrices(init, theta_p, phi_p) -> None:
    """Test gate operations with two parameters with PyQuEST interface"""
    op = init
    matrix_gate = op.unitary_matrix_from_parameters(theta=theta_p, spherical_phi=phi_p)
    if op.number_of_qubits() == 1:
        (theta, spherical_phi, q0) = ('theta', 'spherical_phi', 0)
        operation = op(qubit=q0,
                       theta=theta, spherical_phi=spherical_phi)
    else:
        (theta, spherical_phi, q0, q1) = ('theta', 'spherical_phi', 0, 1)
        operation = op(control=q1, qubit=q0,
                       theta=theta, spherical_phi=spherical_phi)

    if op.number_of_qubits() == 1:
        operation.substitute_parameters({'theta': theta_p,
                                         'spherical_phi': phi_p})
    else:
        operation.substitute_parameters({'theta': theta_p,
                                         'spherical_phi': phi_p})

    if matrix_gate.shape == (2, 2):
        matrix_reconstructed = build_one_qubit_matrix(operation, {})
        matrix_reconstructed_Xmon = build_one_qubit_matrix_Xmon(operation, {})
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed = build_two_qubit_matrix(operation, {})
        matrix_reconstructed_Xmon = build_two_qubit_matrix_Xmon(operation, {})
    matrix_to_test = (matrix_gate @ matrix_reconstructed_Xmon.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)


@pytest.mark.parametrize("init", [ops.PMInteraction,
                                  ])
@pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_PM(init, theta) -> None:
    """Test plus-minus gate operation with PyQuEST interface"""
    op = init
    matrix_gate = op.unitary_matrix_from_parameters(theta)
    if op.number_of_qubits() == 1:
        (Theta, q0) = ('theta', 0)
        operation = op(qubit=q0,
                       theta=Theta,)
    else:
        (Theta, q0, q1) = ('theta', 0, 1)
        operation = op(i=q1, j=q0,
                       theta=Theta)

    if op.number_of_qubits() == 1:
        operation.substitute_parameters({'theta': theta,
                                         })
    else:
        operation.substitute_parameters({'theta': theta,
                                         })

    if matrix_gate.shape == (2, 2):
        matrix_reconstructed_Xmon = build_one_qubit_matrix_Xmon(operation, {})
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed_Xmon = build_two_qubit_matrix_Xmon(operation, {})
    matrix_to_test = (matrix_gate @ matrix_reconstructed_Xmon.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))


@pytest.mark.parametrize("init", [ops.GivensRotation,
                                  ops.GivensRotationLittleEndian,
                                  ])
@pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
@pytest.mark.parametrize("phi", list(np.arange(2 * np.pi / 3, 2 * np.pi, 2 * np.pi / 3)))
def test_Givens(init, theta, phi) -> None:
    """Test Givens rotation gate operation with PyQuEST interface"""
    op = init
    matrix_gate = op.unitary_matrix_from_parameters(theta, phi)
    if op.number_of_qubits() == 1:
        (theta, phi, q0) = ('theta',
                            'phi', 0)
        operation = op(qubit=q0,
                       theta=theta, phi=phi)
    else:
        (Theta, Phi, q0, q1) = ('theta',
                                'phi', 0, 1)
        if init == ops.GivensRotation:
            operation = op(qubit=q0, control=q1,
                           theta=Theta, phi=Phi)
        else:
            operation = op(qubit=q0, control=q1,
                           theta=Theta, phi=Phi)

    if op.number_of_qubits() == 1:
        operation.substitute_parameters({'theta': theta,
                                         'phi': phi})
    else:
        operation.substitute_parameters({'theta': theta,
                                         'phi': phi})

    if matrix_gate.shape == (2, 2):
        matrix_reconstructed_Xmon = build_one_qubit_matrix_Xmon(operation, {})
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed_Xmon = build_two_qubit_matrix_Xmon(operation, {})
    matrix_to_test = (matrix_gate @ matrix_reconstructed_Xmon.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))


@pytest.mark.parametrize("init", [ops.Bogoliubov,
                                  ])
@pytest.mark.parametrize("delta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
@pytest.mark.parametrize("delta_arg", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_Bogoliubov(init, delta, delta_arg) -> None:
    """Test Bogoliubov-deGennes gate operation with PyQuEST interface"""
    delta_real = np.real(delta * np.exp(1j * delta_arg))
    delta_imag = np.imag(delta * np.exp(1j * delta_arg))
    op = init
    matrix_gate = op.unitary_matrix_from_parameters(delta_real, delta_imag)
    if op.number_of_qubits() == 1:
        (Delta_real, Delta_imag, q0) = ('Delta_real',
                                        'Delta_imag', 0)
        operation = op(qubit=q0,
                       Delta_real=Delta_real, Delta_imag=Delta_imag)
    else:
        (Delta_real, Delta_imag, q0, q1) = ('Delta_real',
                                            'Delta_imag', 0, 1)
        operation = op(i=q1, j=q0,
                       Delta_real=Delta_real, Delta_imag=Delta_imag)

    if op.number_of_qubits() == 1:
        operation.substitute_parameters({'Delta_real': delta_real,
                                         'Delta_imag': delta_imag})
    else:
        operation.substitute_parameters({'Delta_real': delta_real,
                                         'Delta_imag': delta_imag})

    if matrix_gate.shape == (2, 2):
        matrix_reconstructed_Xmon = build_one_qubit_matrix_Xmon(operation, {})
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed_Xmon = build_two_qubit_matrix_Xmon(operation, {})
    matrix_to_test = (matrix_gate @ matrix_reconstructed_Xmon.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))


parameter_list = [0, .1, np.pi, -np.pi, np.pi / 4,
                  2 * np.pi, -np.pi - .1, -.1, np.pi + .1, 2 * np.pi + .1]
parameter_list3 = [[p1, p2, p2] for p1 in parameter_list for p2 in parameter_list[:3]] + \
                  [[p2, p1, p2] for p1 in parameter_list[3:] for p2 in parameter_list[:3]] + \
                  [[p2, p2, p1] for p1 in parameter_list[3:] for p2 in parameter_list[:3]]


@pytest.mark.parametrize("init", [ops.Fsim])
@pytest.mark.parametrize("U, t, Delta", parameter_list3)
def test_Fsim(init, U, t, Delta) -> None:
    """Test fermionic simulation gate operation with PyQuEST interface"""
    op = init
    matrix_gate = op.unitary_matrix_from_parameters(U, t, Delta)
    if op.number_of_qubits() == 1:
        (Us, ts, Deltas, q0) = ('U', 't', 'Delta', 0)
        operation = op(qubit=q0,
                       U=Us, t=ts, Delta=Deltas)
    else:
        (Us, ts, Deltas, q0, q1) = ('U', 't', 'Delta', 0, 1)
        operation = op(qubit=q0, control=q1,
                       U=Us, t=ts, Delta=Deltas)

    if op.number_of_qubits() == 1:
        operation.substitute_parameters({'U': U,
                                         't': t, 'Delta': Delta})
    else:
        operation.substitute_parameters({'U': U,
                                         't': t, 'Delta': Delta})

    if matrix_gate.shape == (2, 2):
        matrix_reconstructed_Xmon = build_one_qubit_matrix_Xmon(operation, {})
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed_Xmon = build_two_qubit_matrix_Xmon(operation, {})
    matrix_to_test = (matrix_gate @ matrix_reconstructed_Xmon.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))


@pytest.mark.parametrize("init", [ops.Qsim])
@pytest.mark.parametrize("x, y, z", parameter_list3)
def test_Qsim(init, x, y, z) -> None:
    """Test spin swap simulation gate operation with PyQuEST interface"""
    op = init
    matrix_gate = op.unitary_matrix_from_parameters(x, y, z)
    if op.number_of_qubits() == 1:
        (Us, ts, Deltas, q0) = ('x', 'y', 'z', 0)
        operation = op(qubit=q0,
                       U=Us, t=ts, Delta=Deltas)
    else:
        (xs, ys, zs, q0, q1) = ('x', 'y', 'z', 0, 1)
        operation = op(qubit=q0, control=q1,
                       x=xs, y=ys, z=zs)

    if op.number_of_qubits() == 1:
        operation.substitute_parameters({'x': x,
                                         'y': y, 'z': z})
    else:
        operation.substitute_parameters({'x': x,
                                         'y': y, 'z': z})

    if matrix_gate.shape == (2, 2):
        matrix_reconstructed_Xmon = build_one_qubit_matrix_Xmon(operation, {})
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed_Xmon = build_two_qubit_matrix_Xmon(operation, {})
    matrix_to_test = (matrix_gate @ matrix_reconstructed_Xmon.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))


def build_one_qubit_matrix(operation, gate_args,
                           substitution_dict: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Return full process tomography on single qubit operation with PyQuEST interface"""
    matrix = np.zeros((2, 2), dtype=complex)
    for co, state in enumerate([np.array([1, 0]),
                                np.array([0, 1])]):
        env = utils.createQuestEnv()()
        qubits = utils.createQureg()(1, env)
        cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
        circuit = Circuit()
        circuit += operation
        if substitution_dict is None:
            calculator = None
        else:
            calculator = Calculator()
            for name, val in substitution_dict.items():
                calculator.set(name, val)
        pyquest_call_circuit(circuit=circuit,
                             qureg=qubits,
                             classical_registers={},
                             calculator=calculator)
        for i in range(0, 2):
            out = cheat.getAmp()(qureg=qubits, index=i)
            matrix[i, co] = out
        utils.destroyQureg()(qubits, env=env)
        utils.destroyQuestEnv()(env)
    return matrix


def build_two_qubit_matrix(operation, gate_args,
                           substitution_dict: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Return full process tomography on two qubit operation with PyQuEST interface"""
    matrix = np.zeros((4, 4), dtype=complex)
    for co, state in enumerate([np.array([1, 0, 0, 0]),
                                np.array([0, 1, 0, 0]),
                                np.array([0, 0, 1, 0]),
                                np.array([0, 0, 0, 1]), ]):
        env = utils.createQuestEnv()()
        qubits = utils.createQureg()(2, env)
        cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
        circuit = Circuit()
        circuit += operation
        if substitution_dict is None:
            calculator = None
        else:
            calculator = Calculator()
            for name, val in substitution_dict.items():
                calculator.set(name, val)
        pyquest_call_circuit(circuit=circuit,
                             qureg=qubits,
                             classical_registers={},
                             calculator=calculator)
        for i in range(0, 4):
            out = cheat.getAmp()(qureg=qubits, index=i)
            matrix[i, co] = out
        utils.destroyQureg()(qubits, env=env)
        utils.destroyQuestEnv()(env)
    return matrix


def build_one_qubit_matrix_Xmon(operation, gate_args,
                                substitution_dict: Optional[Dict[str, float]] = None
                                ) -> np.ndarray:
    """Return full process tomography on single qubit operation with PyQuEST interface"""
    matrix = np.zeros((2, 2), dtype=complex)
    for co, state in enumerate([np.array([1, 0]),
                                np.array([0, 1])]):
        env = utils.createQuestEnv()()
        qubits = utils.createQureg()(1, env)
        cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
        circuit = Circuit()
        circuit += operation
        if substitution_dict is None:
            calculator = None
        else:
            calculator = Calculator()
            for name, val in substitution_dict.items():
                calculator.set(name, val)
        pyquest_call_circuit(circuit=circuit,
                             qureg=qubits,
                             classical_registers={},
                             calculator=calculator)
        for i in range(0, 2):
            out = cheat.getAmp()(qureg=qubits, index=i)
            matrix[i, co] = out
        utils.destroyQureg()(qubits, env=env)
        utils.destroyQuestEnv()(env)
    return matrix


def build_two_qubit_matrix_Xmon(operation, gate_args,
                                substitution_dict: Optional[Dict[str, float]] = None
                                ) -> np.ndarray:
    """Return full process tomography on two qubit operation with PyQuEST interface"""
    matrix = np.zeros((4, 4), dtype=complex)
    for co, state in enumerate([np.array([1, 0, 0, 0]),
                                np.array([0, 1, 0, 0]),
                                np.array([0, 0, 1, 0]),
                                np.array([0, 0, 0, 1]), ]):
        env = utils.createQuestEnv()()
        qubits = utils.createQureg()(2, env)
        cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
        circuit = Circuit()
        circuit += operation
        if substitution_dict is None:
            calculator = None
        else:
            calculator = Calculator()
            for name, val in substitution_dict.items():
                calculator.set(name, val)
        pyquest_call_circuit(circuit=circuit,
                             qureg=qubits,
                             classical_registers={},
                             calculator=calculator)
        for i in range(0, 4):
            out = cheat.getAmp()(qureg=qubits, index=i)
            matrix[i, co] = out
        utils.destroyQureg()(qubits, env=env)
        utils.destroyQuestEnv()(env)
    return matrix


if __name__ == '__main__':
    pytest.main(sys.argv)
