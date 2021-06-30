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
"""PyQuEST Backend"""


import pyquest_cffi
from qoqo import Circuit
from typing import (
    Tuple,
    List,
    Optional,
    Dict,
    Any,
    cast
)
from qoqo_pyquest import pyquest_call_circuit


class PyQuestBackend(object):
    r"""Interactive PyQuEST backend to qoqo.

    The PyQuEST backend to qoqo translates the circuit we want to simulate using the qoqo_PyQuEST
    interface, and simulates it using PyQuEST. The results are then returned in the form of
    classical registers, accessible through the results dictionary.
    """

    def __init__(self,
                 number_qubits: int = 1,
                 device: Optional[Any] = None,
                 repetitions: int = 1,
                 ) -> None:
        """Initialize PyQuEST backend

        Args:
            number_qubits: The number of qubits to use
            device: The device specification
            repetitions: The number of repetitions the circuit is run repeatedly,
                         only used when random Paulis
                         or statistical overrotations are in the circuit
        """
        self.name = "pyquest_cffi"
        self.number_qubits = number_qubits
        self.device = device
        self.repetitions = repetitions

    def run_circuit(self, circuit: Circuit
                    ) -> Tuple[Dict[str, List[List[bool]]],
                               Dict[str, List[List[float]]],
                               Dict[str, List[List[complex]]]]:
        """Run a circuit with the PyQuEST backend

        Args:
            circuit: The circuit that is run

        Returns:
            Union[None, Dict[str, 'RegisterOutput']]

        """
        # Initializing the classical registers for calculation and output
        internal_bit_register_dict: Dict[str, List[bool]] = dict()
        internal_float_register_dict: Dict[str, List[float]] = dict()
        internal_complex_register_dict: Dict[str, List[complex]] = dict()

        output_bit_register_dict: Dict[str, List[List[bool]]] = dict()
        output_float_register_dict: Dict[str, List[List[float]]] = dict()
        output_complex_register_dict: Dict[str, List[List[complex]]] = dict()

        for bit_def in circuit.filter_by_tag("DefinitionBit"):
            internal_bit_register_dict[bit_def.name()] = [False for _ in range(bit_def.length())]
            if bit_def.is_output():
                output_bit_register_dict[bit_def.name()] = list()

        for float_def in circuit.filter_by_tag("DefinitionFloat"):
            internal_float_register_dict[float_def.name()] = [
                0.0 for _ in range(float_def.length())]
            if float_def.is_output():
                output_float_register_dict[float_def.name()] = cast(List[List[float]], list())

        for complex_def in circuit.filter_by_tag("DefinitionComplex"):
            internal_complex_register_dict[complex_def.name()] = [
                complex(0.0) for _ in range(complex_def.length())]
            if complex_def.is_output():
                output_complex_register_dict[complex_def.name()] = cast(List[List[complex]], list())

        global_phase = 0
        for op in circuit.filter_by_tag('PragmaGlobalPhase'):
            global_phase += op.phase().float()

        env = pyquest_cffi.utils.createQuestEnv()()
        number_gates_requiring_repetitions = circuit.count_occurences(
            ["PragmaRandomNoise", "PragmaOverrotation"])
        if number_gates_requiring_repetitions > 0:
            repetitions = self.repetitions
        else:
            repetitions = 1
        number_qubits = self.number_qubits
        for _ in range(repetitions):
            # Resetting internat classical registers
            for bit_def in circuit.filter_by_tag("DefinitionBit"):
                internal_bit_register_dict[bit_def.name()] = [
                    False for _ in range(bit_def.length())]

            for float_def in circuit.filter_by_tag("DefinitionFloat"):
                internal_float_register_dict[float_def.name()] = [
                    0.0 for _ in range(float_def.length())]

            for complex_def in circuit.filter_by_tag("DefinitionComplex"):
                internal_complex_register_dict[complex_def.name()] = [
                    complex(0.0) for _ in range(complex_def.length())]

            # Count gates that require density matrix mode
            # Note: Random noise does not, because it is a stochastic unravelling
            number_gate_requiring_density_matrix_mode = circuit.count_occurences(
                ["PragmaDamping", "PragmaDephasing", "PragmaDepolarising"])

            if number_gate_requiring_density_matrix_mode > 0:
                self.qureg = pyquest_cffi.utils.createDensityQureg()(
                    num_qubits=number_qubits,
                    env=env
                )
            else:
                self.qureg = pyquest_cffi.utils.createQureg()(
                    num_qubits=number_qubits,
                    env=env
                )
            if number_gates_requiring_repetitions > 0:
                circuit = circuit.overrotate()
            pyquest_call_circuit(
                circuit=circuit,
                qureg=self.qureg,
                classical_bit_registers=internal_bit_register_dict,
                classical_float_registers=internal_float_register_dict,
                classical_complex_registers=internal_complex_register_dict,
                output_bit_register_dict=output_bit_register_dict,
            )
            for name, reg in output_bit_register_dict.items():
                if name in internal_bit_register_dict.keys():
                    reg.append(internal_bit_register_dict[name])

            for name, reg in output_float_register_dict.items():  # type: ignore
                if name in internal_float_register_dict.keys():
                    reg.append(internal_float_register_dict[name])  # type: ignore

            for name, reg in output_complex_register_dict.items():  # type: ignore
                if name in internal_complex_register_dict.keys():
                    reg.append(internal_complex_register_dict[name])    # type: ignore

            # Overwriting global phase
            if 'global_phase' in output_float_register_dict.keys() and global_phase != 0:
                output_float_register_dict['global_phase'] = [[global_phase]]

        pyquest_cffi.utils.destroyQuestEnv()(env)

        return (output_bit_register_dict, output_float_register_dict, output_complex_register_dict)

    def run_measurement_registers(self, measurement: Any
                                  ) -> Tuple[Dict[str, List[List[bool]]],
                                             Dict[str, List[List[float]]],
                                             Dict[str, List[List[complex]]]]:
        """Run a all circuits of a measurement with the PyQuEST backend

        Args:
            measurement: The measurement that is run

        Returns:
            Union[None, Dict[str, 'RegisterOutput']]

        """
        # Initializing the classical registers for calculation and output

        constant_circuit = measurement.constant_circuit()
        output_bit_register_dict: Dict[str, List[List[bool]]] = dict()
        output_float_register_dict: Dict[str, List[List[float]]] = dict()
        output_complex_register_dict: Dict[str, List[List[complex]]] = dict()
        for circuit in measurement.circuits():
            if constant_circuit is None:
                run_circuit = circuit
            else:
                run_circuit = constant_circuit + circuit

            (tmp_bit_register_dict,
             tmp_float_register_dict,
             tmp_complex_register_dict) = self.run_circuit(
                run_circuit
            )
            output_bit_register_dict.update(tmp_bit_register_dict)
            output_float_register_dict.update(tmp_float_register_dict)
            output_complex_register_dict.update(tmp_complex_register_dict)
        return (
            output_bit_register_dict,
            output_float_register_dict,
            output_complex_register_dict)

    def run_measurement(self, measurement: Any
                        ) -> Optional[Dict[str, float]]:
        """Run a circuit with the PyQuEST backend

        Args:
            measurement: The measurement that is run

        Returns:
            Union[None, Dict[str, 'RegisterOutput']]

        """
        # Initializing the classical registers for calculation and output

        (output_bit_register_dict,
            output_float_register_dict,
            output_complex_register_dict) = self.run_measurement_registers(measurement)
        return measurement.evaluate(
            output_bit_register_dict,
            output_float_register_dict,
            output_complex_register_dict)
