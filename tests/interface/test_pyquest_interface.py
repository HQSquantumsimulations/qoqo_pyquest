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
from typing import (
    Dict,
    Optional
)
from copy import copy
from qoqo_pyquest import (
    pyquest_call_operation,
    pyquest_call_circuit
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
])
def test_simple_single_qubit_gate_matrices(init):
    """Test gate operations without free parameters with PyQuEST interface"""
    op = init
    q0 = 0
    operation = op(qubit=q0)
    matrix_gate = operation.unitary_matrix()

    # Testing gate functionality with pyquest

    matrix_reconstructed = build_one_qubit_matrix(operation)

    if matrix_reconstructed is not None:
        npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)
    matrix_to_test = (matrix_gate @ matrix_reconstructed.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))


@pytest.mark.parametrize("init", [
    ops.CNOT,
    ops.SWAP,
    ops.ControlledPauliZ,
    ops.ControlledPauliY,
    ops.MolmerSorensenXX,
    ops.InvSqrtISwap,
    ops.SqrtISwap,
])
def test_simple_two_qubit_gate_matrices(init):
    """Test gate operations without free parameters with PyQuEST interface"""
    op = init
    q0, q1 = (0, 1)
    operation = op(control=q1, target=q0)
    matrix_gate = operation.unitary_matrix()

    # Testing gate functionality with pyquest

    matrix_reconstructed = build_two_qubit_matrix(operation)
    if matrix_reconstructed is not None:
        npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)
    matrix_to_test = (matrix_gate @ matrix_reconstructed.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))


@pytest.mark.parametrize("init", [
    (ops.RotateX, {'theta': -np.pi, }),
    (ops.RotateY, {'theta': -np.pi, }),
    (ops.RotateZ, {'theta': -np.pi, }),
    #(ops.RotateAroundSphericalAxis, {'theta': -np.pi, 'spherical_theta': np.pi, 'spherical_phi': np.pi,
    #                                 }),
    (ops.SingleQubitGate, {
        'alpha_r': 1, 'alpha_i': 0, 'beta_r': 0, 'beta_i': 0, 'global_phase': 0}),
])
def test_parameterised_sinlge_qubit_gate_matrices(init):
    """Test gate operations with free parameters with PyQuEST interface"""
    op = init[0]
    operation = op(qubit=0, **init[1])
    matrix_gate = operation.unitary_matrix()

    # Testing gate functionality with pyquest
    matrix_reconstructed = build_one_qubit_matrix(operation)
    if matrix_reconstructed is not None:
        npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)
    matrix_to_test = (matrix_gate @ matrix_reconstructed.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))

# @pytest.mark.parametrize("init", [
#     (ops.ControlledPhaseShift, {'theta': -np.pi}),
# ])
# def test_parameterised_two_qubit_gate_matrices(init):
#     """Test gate operations with free parameters with PyQuEST interface"""
#     op = init[0]

#     q0, q1 = (0, 1)
#     operation = op(control=q1, target=q0, **init[1])
#     matrix_gate = operation.unitary_matrix()

#     # Testing gate functionality with pyquest

#     matrix_reconstructed = build_two_qubit_matrix(operation)
#     if matrix_reconstructed is not None:
#         npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)
#     matrix_to_test = (matrix_gate @ matrix_reconstructed.conj().T)
#     matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
#     npt.assert_almost_equal(
#         (matrix_to_test),
#         np.identity(matrix_gate.shape[0]))


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

    operation = op(qubit=0, alpha_r=np.real(alpha), alpha_i=np.imag(alpha),
                   beta_r=np.real(beta), beta_i=np.imag(beta), global_phase=d)
    matrix_gate = operation.unitary_matrix()


    matrix_reconstructed = build_one_qubit_matrix(operation)

    matrix_to_test = (matrix_gate @ matrix_reconstructed.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))


@pytest.mark.parametrize("init", [ops.RotateX,
                                  ops.RotateY,
                                  ops.RotateZ,
                                  ])
@pytest.mark.parametrize("theta_p", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_single_parameter_gate_matrices(init, theta_p) -> None:
    """Test gate operations with single parameter with PyQuEST interface"""
    op = init

    operation = op(qubit=0, theta=theta_p)
    matrix_reconstructed = build_one_qubit_matrix(operation)

    matrix_gate = operation.unitary_matrix()
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)

    matrix_to_test = (matrix_gate @ matrix_reconstructed.conj().T)
    matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
    npt.assert_almost_equal(
        (matrix_to_test),
        np.identity(matrix_gate.shape[0]))
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)


# @pytest.mark.parametrize("init", [
# #                                  ops.ControlledPhaseShift,
#                                   ops.PMInteraction
#                                   ])
# @pytest.mark.parametrize("theta_p", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
# def test_single_parameter_two_qubit_gate_matrices(init, theta_p) -> None:
#     """Test gate operations with single parameter with PyQuEST interface"""
#     op = init

#     operation = op(control=1, target=0, theta=theta_p)

#     matrix_reconstructed = build_two_qubit_matrix(operation)
#     matrix_gate = operation.unitary_matrix()
#     npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)

#     matrix_to_test = (matrix_gate @ matrix_reconstructed.conj().T)
#     matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
#     npt.assert_almost_equal(
#         (matrix_to_test),
#         np.identity(matrix_gate.shape[0]))
#     npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)


# @pytest.mark.parametrize("init", [ops.GivensRotation,
#                                   ops.GivensRotationLittleEndian,
#                                   ])
# @pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
# @pytest.mark.parametrize("phi", list(np.arange(2 * np.pi / 3, 2 * np.pi, 2 * np.pi / 3)))
# def test_Givens(init, theta, phi) -> None:
#     """Test Givens rotation gate operation with PyQuEST interface"""
#     op = init

#     if init == ops.GivensRotation:
#         operation = op(target=0, control=1,
#                         theta=theta, phi=phi)
#     else:
#         operation = op(target=0, control=1,
#                         theta=theta, phi=phi)
#     matrix_gate = operation.unitary_matrix()
    
#     matrix_reconstructed = build_two_qubit_matrix(operation)
#     matrix_to_test = (matrix_gate @ matrix_reconstructed.conj().T)
#     matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
#     npt.assert_almost_equal(
#         (matrix_to_test),
#         np.identity(matrix_gate.shape[0]))


# @pytest.mark.parametrize("init", [ops.Bogoliubov,
#                                   ])
# @pytest.mark.parametrize("delta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
# @pytest.mark.parametrize("delta_arg", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
# def test_Bogoliubov(init, delta, delta_arg) -> None:
#     """Test Bogoliubov-deGennes gate operation with PyQuEST interface"""
#     delta_real = np.real(delta * np.exp(1j * delta_arg))
#     delta_imag = np.imag(delta * np.exp(1j * delta_arg))
#     op = init
   
#     operation = op(control=0, target=1,
#                        Delta_real=delta_real, Delta_imag=delta_imag)
#     matrix_gate = operation.unitary_matrix()


#     matrix_reconstructed = build_two_qubit_matrix(operation,)
#     matrix_to_test = (matrix_gate @ matrix_reconstructed.conj().T)
#     matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
#     npt.assert_almost_equal(
#         (matrix_to_test),
#         np.identity(matrix_gate.shape[0]))


# parameter_list = [0, .1, np.pi, -np.pi, np.pi / 4,
#                   2 * np.pi, -np.pi - .1, -.1, np.pi + .1, 2 * np.pi + .1]
# parameter_list3 = [[p1, p2, p2] for p1 in parameter_list for p2 in parameter_list[:3]] + \
#                   [[p2, p1, p2] for p1 in parameter_list[3:] for p2 in parameter_list[:3]] + \
#                   [[p2, p2, p1] for p1 in parameter_list[3:] for p2 in parameter_list[:3]]


# @pytest.mark.parametrize("init", [ops.Fsim])
# @pytest.mark.parametrize("U, t, Delta", parameter_list3)
# def test_Fsim(init, U, t, Delta) -> None:
#     """Test fermionic simulation gate operation with PyQuEST interface"""
#     op = init
#     operation = op(target=9, control=1,
#                        U=U, t=t, Delta=Delta)
#     matrix_gate = operation.unitary_matrix()
    
#     matrix_reconstructed = build_two_qubit_matrix(operation)
#     matrix_to_test = (matrix_gate @ matrix_reconstructed.conj().T)
#     matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
#     npt.assert_almost_equal(
#         (matrix_to_test),
#         np.identity(matrix_gate.shape[0]))


# @pytest.mark.parametrize("init", [ops.Qsim])
# @pytest.mark.parametrize("x, y, z", parameter_list3)
# def test_Qsim(init, x, y, z) -> None:
#     """Test spin swap simulation gate operation with PyQuEST interface"""
#     op = init
#     operation = op(target=0, control=1,
#                        x=x, y=y, z=z)
#     matrix_gate = operation.unitary_matrix()
   
#     matrix_reconstructed = build_two_qubit_matrix(operation)
#     matrix_to_test = (matrix_gate @ matrix_reconstructed.conj().T)
#     matrix_to_test = matrix_to_test / matrix_to_test[0, 0]
#     npt.assert_almost_equal(
#         (matrix_to_test),
#         np.identity(matrix_gate.shape[0]))


def build_one_qubit_matrix(operation) -> np.ndarray:
    """Return full process tomography on single qubit operation with PyQuEST interface"""
    matrix = np.zeros((2, 2), dtype=complex)
    for co, state in enumerate([np.array([1, 0]),
                                np.array([0, 1])]):
        env = utils.createQuestEnv()()
        qubits = utils.createQureg()(1, env)
        cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
        circuit = Circuit()
        circuit += operation
        pyquest_call_circuit(circuit=circuit,
                             qureg=qubits,
                             classical_bit_registers=dict(),
                             classical_float_registers=dict(),
                             classical_complex_registers=dict(),
                             output_bit_register_dict=dict(),
                             )
        for i in range(0, 2):
            out = cheat.getAmp()(qureg=qubits, index=i)
            matrix[i, co] = out
        utils.destroyQureg()(qubits, env=env)
        utils.destroyQuestEnv()(env)
    return matrix


def build_two_qubit_matrix(operation) -> np.ndarray:
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
        
        pyquest_call_circuit(circuit=circuit,
                             qureg=qubits,
                             classical_bit_registers=dict(),
                             classical_float_registers=dict(),
                             classical_complex_registers=dict(),
                             output_bit_register_dict=dict(),)
        for i in range(0, 4):
            out = cheat.getAmp()(qureg=qubits, index=i)
            matrix[i, co] = out
        utils.destroyQureg()(qubits, env=env)
        utils.destroyQuestEnv()(env)
    return matrix



if __name__ == '__main__':
    pytest.main(sys.argv)
