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
"""Test qoqo pyquest interface for PRAGMAs"""
import pytest
import sys
import numpy as np
import numpy.testing as npt
from qoqo import operations as ops
from qoqo import Circuit
from pyquest_cffi import cheat
from pyquest_cffi import utils
from pyquest_cffi import ops as qops

from copy import copy
from qoqo_pyquest import pyquest_call_operation, pyquest_call_circuit


def test_set_qureg():
    """Test set qureg PRAGMA with PyQuEST interface"""
    env = utils.createQuestEnv()()
    qureg_density = utils.createDensityQureg()(2, env)
    qureg_wave = utils.createQureg()(2, env)
    cheat.initBlankState()(qureg=qureg_density)
    cheat.initBlankState()(qureg=qureg_wave)

    density_start = cheat.getDensityMatrix()(qureg_density)
    wave_start = cheat.getStateVector()(qureg_wave)

    set_wave = np.array([2, 1, 0, 1])
    set_density = np.array([[1, 0, 2, 0],
                            [0, 2, 3, 3],
                            [3, 1, 0, 0],
                            [0, 1, 2, 0]])

    pyquest_call_operation(operation=ops.PragmaSetStateVector(set_wave),
                           qureg=qureg_wave,
                           classical_bit_registers=dict(),
                           classical_float_registers=dict(),
                           classical_complex_registers=dict(),
                           output_bit_register_dict=dict(),)
    pyquest_call_operation(operation=ops.PragmaSetDensityMatrix(set_density.flatten()),
                           qureg=qureg_density,
                           classical_bit_registers=dict(),
                           classical_float_registers=dict(),
                           classical_complex_registers=dict(),
                           output_bit_register_dict=dict(),)

    npt.assert_array_equal(density_start, [[0] * 4] * 4)
    npt.assert_array_equal(wave_start, [0] * 4)
    npt.assert_array_equal(set_wave, cheat.getStateVector()(qureg_wave))
    npt.assert_array_equal(set_density, cheat.getDensityMatrix()(qureg_density))

    utils.destroyQureg().call_interactive(qureg_wave, env)
    utils.destroyQureg().call_interactive(qureg_density, env)
    utils.destroyQuestEnv().call_interactive(env)


def test_active_reset():
    """Test ActiveReset PRAGMA with PyQuEST interface"""
    definition = ops.DefinitionBit(name='ro', length=2, is_output=True)
    circuit = Circuit()
    circuit += definition
    circuit += ops.PauliX(qubit=0)

    circuit_with = circuit + ops.PragmaActiveReset(qubit=0) + ops.PragmaActiveReset(qubit=1)
    circuit_without = circuit + ops.PauliX(qubit=0)
    circuit_list = [circuit, circuit_with, circuit_without]
    for circuit in circuit_list:
        circuit += ops.MeasureQubit(qubit=0, readout='ro', readout_index=0)
        circuit += ops.MeasureQubit(qubit=1, readout='ro', readout_index=1)

    test_dict = {'ro': [False, False]}
    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(2, env)
    results_list_0 = []
    results_list_1 = []
    for circuit in circuit_list:
        pyquest_call_circuit(circuit=circuit, qureg=qureg, classical_bit_registers=test_dict,
                             classical_float_registers=dict(),
                             classical_complex_registers=dict(),
                             output_bit_register_dict=dict(),)
        results_list_0.append(test_dict['ro'][0])
        results_list_1.append(test_dict['ro'][1])

    assert results_list_0 == [True, False, False]
    assert results_list_1 == [False, False, False]


@pytest.mark.parametrize("init", [(ops.PragmaDamping,
                                   'PragmaDamping(gate_time, rate) 0',
                                   qops.mixDamping()),
                                  (ops.PragmaDepolarising,
                                   'PragmaDepolarising(gate_time, rate) 0',
                                   qops.mixDepolarising()),
                                  (ops.PragmaDephasing,
                                   'PragmaDephasing(gate_time, rate) 0',
                                   qops.mixDephasing())
                                  ])
@pytest.mark.parametrize("gate_time", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
@pytest.mark.parametrize("rate", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
def test_noise_operators(init, gate_time, rate):
    """Test PRAGMA operators applying noise with PyQuEST interface"""
    op = init[0]

    (Gate_time, Rate, q0) = ('gate_time',
                             'rate', 0)
    operation = op(qubit=q0, gate_time=Gate_time, rate=Rate)
    substitution_dict = {'gate_time': gate_time, 'rate': rate}
    operation = operation.substitute_parameters(substitution_dict)
    env = utils.createQuestEnv()()
    qureg = utils.createDensityQureg()(1, env)
    cheat.initPlusState()(qureg=qureg)
    init[2](qureg=qureg, probability=operation.probability().float(), qubit=0)
    density_matrix_ref = cheat.getDensityMatrix()(qureg)
    test_dict = {'ro': list()}

    cheat.initPlusState()(qureg)
    pyquest_call_operation(operation=operation.substitute_parameters(substitution_dict),
                           qureg=qureg,
                           classical_bit_registers=dict(),
                           classical_float_registers=dict(),
                           classical_complex_registers=test_dict,
                           output_bit_register_dict=dict(),
                           )

    density_matrix_test = cheat.getDensityMatrix()(qureg)

    npt.assert_array_almost_equal(density_matrix_ref, density_matrix_test)

    utils.destroyQureg().call_interactive(qureg, env)
    utils.destroyQuestEnv().call_interactive(env)

    operation = operation.substitute_parameters(substitution_dict)

    env = utils.createQuestEnv()()
    qureg = utils.createDensityQureg()(1, env)
    cheat.initPlusState()(qureg=qureg)
    init[2](qureg=qureg, probability=operation.probability().float(), qubit=0)
    density_matrix_ref = cheat.getDensityMatrix()(qureg)

    cheat.initPlusState()(qureg)
    pyquest_call_operation(operation=operation, qureg=qureg, classical_bit_registers=dict(),
                           classical_float_registers=dict(),
                           classical_complex_registers=test_dict,
                           output_bit_register_dict=dict(),)

    density_matrix_test = cheat.getDensityMatrix()(qureg)

    npt.assert_array_almost_equal(density_matrix_ref, density_matrix_test)

    utils.destroyQureg().call_interactive(qureg, env)
    utils.destroyQuestEnv().call_interactive(env)

@pytest.mark.parametrize("gate_time", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
@pytest.mark.parametrize("depolarisation_rate", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
@pytest.mark.parametrize("dephasing_rate", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
def test_random_noise_operator(gate_time, depolarisation_rate, dephasing_rate):
    """Test PRAGMA operators applying random noise (stochastic unravelling) with PyQuEST interface"""
    op = ops.PragmaRandomNoise
    test_dict = {'ro': list()}

    (Gate_time, depol_Rate, dephasing_Rate) = ('gate_time',
                                               'depolarisation_rate',
                                               'dephasing_rate',
                                               )
    operation = op(qubit=0, gate_time=Gate_time,
                   depolarising_rate=depol_Rate, dephasing_rate=dephasing_Rate)

    substitution_dict = {'gate_time': gate_time,
                         'depolarisation_rate': depolarisation_rate,
                         'dephasing_rate': dephasing_rate}
    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(1, env)
    state_vec_ref = cheat.getStateVector()(qureg)
    pyquest_call_operation(operation=operation.substitute_parameters(substitution_dict),
                           qureg=qureg,
                           classical_bit_registers=dict(),
                           classical_float_registers=dict(),
                           classical_complex_registers=test_dict,
                           output_bit_register_dict=dict(),
                           )
    if gate_time == 0 or (dephasing_rate == 0 and depolarisation_rate == 0):
        state_vec = cheat.getStateVector()(qureg)
        npt.assert_array_almost_equal(state_vec_ref, state_vec)

    utils.destroyQureg().call_interactive(qureg, env)
    utils.destroyQuestEnv().call_interactive(env)

    operation.substitute_parameters(substitution_dict)

    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(1, env)
    state_vec_ref = cheat.getStateVector()(qureg)
    pyquest_call_operation(operation=operation.substitute_parameters(substitution_dict),
                           qureg=qureg, classical_bit_registers=dict(),
                           classical_float_registers=dict(),
                           classical_complex_registers=test_dict,
                           output_bit_register_dict=dict(),)
    if gate_time == 0 or (dephasing_rate == 0 and depolarisation_rate == 0):
        state_vec = cheat.getStateVector()(qureg)
        npt.assert_array_almost_equal(state_vec_ref, state_vec)

    utils.destroyQureg().call_interactive(qureg, env)
    utils.destroyQuestEnv().call_interactive(env)


def test_conditional_pragma():
    """Test PRAGMA conditional operation"""
    test_dict = {'ro': [False, False],
                 'cond': [True, False]}

    circuit0 = Circuit()
    circuit0 += ops.PauliX(qubit=0)
    circuit0 += ops.MeasureQubit(readout='ro', qubit=0, readout_index=0)

    circuit1 = Circuit()
    circuit1 += ops.PauliX(qubit=1)
    circuit1 += ops.MeasureQubit(readout='ro', qubit=1, readout_index=1)

    operation = ops.PragmaConditional(
        condition_register='cond',
        condition_index=0,
        circuit=circuit0)

    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(2, env)
    test_dict = {'ro': [False, False],
                 'cond': [True, False]}
    pyquest_call_operation(operation=operation,
                           qureg=qureg,
                           classical_bit_registers=test_dict,
                           classical_float_registers=dict(),
                           classical_complex_registers=dict(),
                           output_bit_register_dict=dict(),
                           )

    utils.destroyQureg().call_interactive(qureg, env)
    utils.destroyQuestEnv().call_interactive(env)

    assert test_dict['ro'] == [True, False]

    operation = ops.PragmaConditional(
        condition_register='cond',
        condition_index=0,
        circuit=circuit1)

    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(2, env)
    test_dict = {'ro': [False, False],
                 'cond': [True, False]}
    pyquest_call_operation(operation=operation,
                           qureg=qureg,
                           classical_bit_registers=test_dict,
                           classical_float_registers=dict(),
                           classical_complex_registers=dict(),
                           output_bit_register_dict=dict(),
                           )

    utils.destroyQureg().call_interactive(qureg, env)
    utils.destroyQuestEnv().call_interactive(env)

    assert test_dict['ro'] == [False, True]

    operation = ops.PragmaConditional(
        condition_register='not_in',
        condition_index=0,
        circuit=circuit1)


    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(2, env)
    with pytest.raises(RuntimeError):
        pyquest_call_operation(operation=operation,
                               qureg=qureg,
                               classical_bit_registers=dict(),
                               classical_float_registers=dict(),
                               classical_complex_registers=test_dict,
                               output_bit_register_dict=dict(),
                               )

    utils.destroyQureg().call_interactive(qureg, env)
    utils.destroyQuestEnv().call_interactive(env)


if __name__ == '__main__':
    pytest.main(sys.argv)
