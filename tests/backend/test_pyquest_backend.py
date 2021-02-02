"""Testing qoqo PyQuEST backend"""
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
from qoqo_pyquest import (
    PyQuestBackend,
    pyquest_call_circuit
)
import os
import json
from hqsbase.calculator import (
    Calculator
)
from hqsbase.qonfig import Qonfig
from pyquest_cffi import utils, cheat


def test_pyquest_interactive_statevector():
    """Testing with PyQuEST statevector"""
    circuit = Circuit()
    circuit += ops.Definition(name='out', vartype='bit', length=2, is_output=True)
    circuit += ops.Definition(name='out2', vartype='bit', length=2, is_output=True)
    circuit += ops.Definition(name='notout', vartype='bit', length=2, is_output=False)
    circuit += ops.PauliX(qubit=0)
    circuit += ops.MeasureQubit(qubit=0, readout='out', readout_index=0)
    circuit += ops.MeasureQubit(qubit=1, readout='out', readout_index=1)

    pyquest = PyQuestBackend(circuit=circuit,
                             number_qubits=2)
    result = pyquest.run()
    assert result['out'].register == [[True, False]]
    assert result['out'].size == 1
    assert result['out2'].size == 1
    assert result['out2'].register == [[False, False]]
    assert 'notout' not in result.keys()

    config = pyquest.to_qonfig()
    json = config.to_json()
    config2 = Qonfig.from_json(json)
    pyquest2 = config2.to_instance()


def test_pyquest_interactive_density_matrix():
    """Test with PyQuEST density matrix"""
    circuit = Circuit()
    circuit += ops.Definition(name='out', vartype='bit', length=2, is_output=True)
    circuit += ops.Definition(name='out2', vartype='bit', length=2, is_output=True)
    circuit += ops.Definition(name='notout', vartype='bit', length=2, is_output=False)
    circuit += ops.PauliX(qubit=0)
    circuit += ops.MeasureQubit(qubit=0, readout='out')
    circuit += ops.MeasureQubit(qubit=1, readout='out')

    pyquest = PyQuestBackend(circuit=circuit,
                                                 number_qubits=2)
    result = pyquest.run()
    assert result['out'].register == [[True, False]]
    assert result['out'].size == 1
    assert result['out2'].size == 1
    assert 'notout' not in result.keys()


def test_pyquest_global_phase():
    """Test with PyQuEST"""
    circuit = Circuit()
    circuit += ops.Definition(name='out', vartype='bit', length=2, is_output=True)
    circuit += ops.Definition(name='global_phase', vartype='float', length=1, is_output=True)
    circuit += ops.PauliX(qubit=0)
    circuit += ops.PragmaGlobalPhase(phase=1)
    circuit += ops.PragmaGlobalPhase(phase='test')
    circuit += ops.MeasureQubit(qubit=0, readout='out')
    circuit += ops.MeasureQubit(qubit=1, readout='out')
    circuit += ops.PragmaParameterSubstitution(substitution_dict={'test': -2})
    pyquest = PyQuestBackend(circuit=circuit,
                                                 number_qubits=2)
    result = pyquest.run()
    assert result['out'].register == [[True, False]]
    assert result['out'].size == 1
    assert result['global_phase'].register[0][0] == -1


def test_pyquest():
    """Test PyQuEST function"""
    circuit = Circuit()
    circuit += ops.RotateX(qubit=0, theta='theta')
    assert circuit.is_parameterized
    calculator = Calculator()
    calculator.set('theta', np.pi)
    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(1, env)
    pyquest_call_circuit(circuit, qureg, {}, calculator)

    state_vec_test = cheat.getStateVector()(qureg)

    npt.assert_array_almost_equal([0, -1j], state_vec_test)

    utils.destroyQureg().call_interactive(qureg, env)
    utils.destroyQuestEnv().call_interactive(env)
    circuit.substitute_parameters(substitution_dict={'theta': np.pi})
    assert not circuit.is_parameterized
    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(1, env)
    pyquest_call_circuit(circuit, qureg, {})

    state_vec_test = cheat.getStateVector()(qureg)

    npt.assert_array_almost_equal([0, -1j], state_vec_test)

    utils.destroyQureg().call_interactive(qureg, env)
    utils.destroyQuestEnv().call_interactive(env)

if __name__ == '__main__':
    pytest.main(sys.argv)
