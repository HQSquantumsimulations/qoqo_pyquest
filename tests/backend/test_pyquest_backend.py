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
from qoqo import operations as ops
from qoqo import Circuit
from qoqo_pyquest import (
    PyQuestBackend,
    pyquest_call_circuit
)
import os
import json

from pyquest_cffi import utils, cheat


def test_pyquest_interactive_statevector():
    """Testing with PyQuEST statevector"""
    circuit = Circuit()
    circuit += ops.DefinitionBit(name='out',  length=2, is_output=True)
    circuit += ops.DefinitionBit(name='out2',  length=2, is_output=True)
    circuit += ops.DefinitionBit(name='notout', length=2, is_output=False)
    circuit += ops.PauliX(qubit=0)
    circuit += ops.MeasureQubit(qubit=0, readout='out', readout_index=0)
    circuit += ops.MeasureQubit(qubit=1, readout='out', readout_index=1)

    pyquest = PyQuestBackend(number_qubits=2)
    (output_bit_register_dict, output_float_register_dict, output_complex_register_dict) = pyquest.run_circuit(circuit)
    assert output_bit_register_dict['out'] == [[True, False]]
    assert output_bit_register_dict['out2'] == [[False, False]]
    assert 'notout' not in output_bit_register_dict.keys()

    


def test_pyquest_interactive_density_matrix():
    """Test with PyQuEST density matrix"""
    circuit = Circuit()
    circuit += ops.DefinitionBit(name='out', length=2, is_output=True)
    circuit += ops.DefinitionBit(name='out2',  length=2, is_output=True)
    circuit += ops.DefinitionBit(name='notout', length=2, is_output=False)
    circuit += ops.PauliX(qubit=0)
    circuit += ops.MeasureQubit(qubit=0, readout='out', readout_index = 0)
    circuit += ops.MeasureQubit(qubit=1, readout='out', readout_index = 1)

    pyquest = PyQuestBackend(number_qubits=2)
    (output_bit_register_dict, output_float_register_dict, output_complex_register_dict) = pyquest.run_circuit(circuit)
    assert output_bit_register_dict['out'] == [[True, False]]
    assert len(output_bit_register_dict['out2']) == 1
    assert 'notout' not in output_bit_register_dict.keys()


def test_pyquest_global_phase():
    """Test with PyQuEST"""
    circuit = Circuit()
    circuit += ops.DefinitionBit(name='out', length=2, is_output=True)
    circuit += ops.DefinitionFloat(name='global_phase', length=1, is_output=True)
    circuit += ops.PauliX(qubit=0)
    circuit += ops.PragmaGlobalPhase(phase=1)
    circuit += ops.PragmaGlobalPhase(phase='test')
    circuit += ops.MeasureQubit(qubit=0, readout='out', readout_index=0)
    circuit += ops.MeasureQubit(qubit=1, readout='out', readout_index=1)
    circuit += ops.InputSymbolic(name="test", input=-2)
    circuit = circuit.substitute_parameters({})
    pyquest = PyQuestBackend(number_qubits=2)
    (output_bit_register_dict, output_float_register_dict, output_complex_register_dict) = pyquest.run_circuit(circuit)
    assert output_bit_register_dict['out'] == [[True, False]]
    assert len(output_bit_register_dict['out']) == 1
    assert output_float_register_dict['global_phase'][0][0] == -1




if __name__ == '__main__':
    pytest.main(sys.argv)
