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

from qoqo.backends import (
    BackendBaseClass
)
from qoqo.registers import (
    RegisterOutput,
    BitRegister,
    FloatRegister,
    ComplexRegister,
    add_register
)
import pyquest_cffi
from qoqo import operations as ops
from qoqo import Circuit
from typing import (
    Union,
    Optional,
    Dict,
    cast
)
from qoqo.devices import DeviceBaseClass
from hqsbase.calculator import (
    Calculator,
    CalculatorFloat
)
from hqsbase.qonfig import Qonfig, empty
from qoqo_pyquest import pyquest_call_circuit
import numpy as np


class PyQuestBackend(BackendBaseClass):
    r"""Interactive PyQuEST backend to qoqo.

    The PyQuEST backend to qoqo translates the circuit we want to simulate using the qoqo_PyQuEST
    interface, and simulates it using PyQuEST. The results are then returned in the form of
    classical registers, accessible through the results dictionary.
    """

    _qonfig_defaults_dict = {
        'circuit': {'doc': 'The circuit that is run',
                    'default': None},
        'number_qubits': {'doc': 'The number of qubits to use',
                          'default': empty},
        'substitution_dict': {'doc': 'Substitution dictionary used to replace symbolic parameters',
                              'default': None},
        'number_measurements': {'doc': 'Number of measurements, has no effect when '
                                + 'random_pauli_errors is False. When random_pauli_errors is True, '
                                + 'number_measurements sets the number of times the circuit runs.',
                                'default': 1},
        'device': {'doc': 'The device specification',
                   'default': None},
        'use_density_matrix': {'doc': 'Run circuit with density matrix/statevector simulator',
                               'default': False},
        'random_pauli_errors': {'doc': 'The executed circuit contains random noise pragmas and the '
                                + 'circuit needs to run repeatedly to achieve stochastic '
                                + 'unravelling of the decoherence in the time-evolution',
                                'default': False},
        'repetitions': {'doc': 'Overrides number_measurements when random Paulis in the circuit',
                        'default': None},
    }

    def __init__(self,
                 circuit: Optional[Circuit] = None,
                 number_qubits: int = 1,
                 substitution_dict: Optional[Dict[str, float]] = None,
                 number_measurements: int = 1,
                 device: Optional[DeviceBaseClass] = None,
                 use_density_matrix: bool = False,
                 random_pauli_errors: bool = False,
                 repetitions: Optional[int] = None,
                 **kwargs) -> None:
        """Initialize PyQuEST backend

        Args:
            circuit: The circuit that is run on the backend
            number_qubits: The number of qubits to use
            substitution_dict: The substitution dictionary used to replace symbolic parameters
            number_measurements: The number of measurements. This parameter has no effect when
                                 random_pauli_errors is False. When random_pauli_errors is True,
                                 number_measurements sets the number of times the circuit runs.
            device: The device specification
            use_density_matrix: Run circuit with density matrix simulator (True) or statevector
                                simulator (False)
            random_pauli_errors: The executed circuit contains random noise pragmas and the circuit
                                 needs to run repeatedly to achieve stochastic unravelling of the
                                 decoherence in the time-evolution
            repetitions: The number of repetitions overrides number_measurements
                         as number of times the circuit is run repeatedly,
                         when random Paulis are in the circuit
            **kwargs: Additional keyword arguments

        """
        self.name = "pyquest_cffi"

        self.number_qubits = number_qubits
        self.substitution_dict: Optional[Dict[str, float]] = substitution_dict
        self.number_measurements = number_measurements
        self.device = device
        self.use_density_matrix = use_density_matrix
        self.random_pauli_errors = random_pauli_errors
        self.repetitions = self.number_measurements if repetitions is None else repetitions
        self.kwargs = kwargs
        self.qubit_names = getattr(self.device, '_qubit_names', None)
        self.state = None

        super().__init__(circuit=circuit,
                         substitution_dict=self.substitution_dict,
                         device=self.device,
                         number_qubits=number_qubits,
                         **kwargs)

        for _, op in enumerate(self.circuit):
            op = cast(ops.Operation, op)
            if 'PragmaOverrotation' in op._operation_tags:
                op = cast(ops.PragmaOverrotation, op)
                if op._type == 'static':
                    random = np.random.normal(float(op._mean), float(op._variance))
                    if self.substitution_dict is None:
                        self.substitution_dict = dict()
                    self.substitution_dict[op._overrotation_parameter] = random

    @property
    def circuit(self) -> Circuit:
        """Return circuit processed by backend

        Setter:
            circuit (Optional[Circuit]): New circuit, backend instructions in circuit
                                             will be applied to backend when using setter

        Returns:
            Optional[Circuit]
        """
        return self._circuit

    @circuit.setter
    def circuit(self, circuit: Circuit) -> None:
        if circuit is None:
            self._circuit = Circuit()
        else:
            for _, op in enumerate(circuit):
                op = cast(ops.Operation, op)
                if 'Pragma' in op._operation_tags:
                    op = cast(ops.Pragma, op)
                    if op.backend_instruction(backend=self.name) is not None:
                        instruction = op.backend_instruction(backend=self.name)
                        if instruction is not None:
                            for key, val in instruction.items():
                                setattr(self, key, val)
            self._circuit = circuit
            for _, op in enumerate(self.circuit):
                op = cast(ops.Operation, op)
                if 'PragmaOverrotation' in op._operation_tags:
                    op = cast(ops.PragmaOverrotation, op)
                    if op._type == 'static':
                        if self.substitution_dict is None:
                            self.substitution_dict = dict()
                        random = np.random.normal(float(op._mean), float(op._variance))
                        self.substitution_dict[op._overrotation_parameter] = random

    @classmethod
    def from_qonfig(cls,
                    config: Qonfig['PyQuestBackend']
                    ) -> 'PyQuestBackend':
        """Create an Instance from Qonfig

        Args:
            config: Qonfig of class

        Returns:
            PyQuestBackend
        """
        if isinstance(config['device'], Qonfig):
            init_device = config['device'].to_instance()
        else:
            init_device = cast(Optional[DeviceBaseClass], config['device'])
        if isinstance(config['circuit'], Qonfig):
            init_circuit = config['circuit'].to_instance()
        else:
            init_circuit = cast(Optional[Circuit], config['circuit'])
        return cls(circuit=init_circuit,
                   number_qubits=config['number_qubits'],
                   substitution_dict=config['substitution_dict'],
                   number_measurements=config['number_measurements'],
                   device=init_device,
                   use_density_matrix=config['use_density_matrix'],
                   random_pauli_errors=config['random_pauli_errors'],
                   repetitions=config['repetitions'],
                   )

    def to_qonfig(self) -> 'Qonfig[PyQuestBackend]':
        """Create a Qonfig from Instance

        Returns:
            Qonfig[PyQuestBackend]
        """
        config = Qonfig(self.__class__)
        if self._circuit is not None:
            config['circuit'] = self._circuit.to_qonfig()
        else:
            config['circuit'] = self._circuit
        config['number_qubits'] = self.number_qubits
        config['substitution_dict'] = self.substitution_dict
        config['number_measurements'] = self.number_measurements
        if self.device is not None:
            config['device'] = self.device.to_qonfig()
        else:
            config['device'] = self.device
        config['use_density_matrix'] = self.use_density_matrix
        config['random_pauli_errors'] = self.random_pauli_errors
        config['repetitions'] = self.repetitions

        return config

    def run(self, keep_state: bool = False, **kwargs
            ) -> Union[None, Dict[str, 'RegisterOutput']]:
        """Run the PyQuEST backend

        Args:
            keep_state: A special property of pyquest simulation. If True, keep the state from
                        quest backend and save in self.state. Used to read out statevectors
            kwargs: Additional keyword arguments

        Returns:
            Union[None, Dict[str, 'RegisterOutput']]

        """
        # Initializing the classical registers for calculation and output
        internal_register_dict: Dict[str, Union[BitRegister,
                                                FloatRegister, ComplexRegister]] = dict()
        output_register_dict: Dict[str, RegisterOutput] = dict()
        for definition in self.circuit._definitions:
            add_register(internal_register_dict, output_register_dict, definition)
        statistic_sub_dict: Dict[str, float] = dict()

        # Setting up calculator for substitutions
        for _, op in enumerate(self.circuit):
            op = cast(ops.Operation, op)
            if 'PragmaOverrotation' in op._operation_tags:
                op = cast(ops.PragmaOverrotation, op)
                if op._type == 'static':
                    random = np.random.normal(float(op._mean), float(op._variance))
                    statistic_sub_dict[op._overrotation_parameter] = random
        if self.substitution_dict is None and statistic_sub_dict == dict():
            calculator = None
        else:
            calculator = Calculator()
            self.substitution_dict = cast(Dict[str, float], self.substitution_dict)
            for name, val in self.substitution_dict.items():
                calculator.set(name, val)
            for name, val in statistic_sub_dict.items():
                calculator.set(name, val)

        global_phase = 0
        for op in self.circuit:
            op = cast(ops.Operation, op)
            if 'PragmaGlobalPhase' in op._operation_tags:
                op = cast(ops.PragmaGlobalPhase, op)
                if calculator is not None:
                    global_phase += CalculatorFloat(calculator.parse_get(op.phase.value))
                else:
                    op = global_phase + CalculatorFloat(op.phase).value

        env = pyquest_cffi.utils.createQuestEnv()()
        if self.random_pauli_errors:
            repetitions = self.repetitions
        else:
            repetitions = 1
        number_qubits = self.number_qubits
        for _ in range(repetitions):
            # Resetting internat classical registers
            for _, reg in internal_register_dict.items():
                reg.reset()
            if self.use_density_matrix:
                self.qureg = pyquest_cffi.utils.createDensityQureg()(
                    num_qubits=number_qubits,
                    env=env
                )
            else:
                self.qureg = pyquest_cffi.utils.createQureg()(
                    num_qubits=number_qubits,
                    env=env
                )
            pyquest_call_circuit(
                circuit=self.circuit,
                qureg=self.qureg,
                classical_registers=internal_register_dict,
                calculator=calculator,
                qubit_names=self.qubit_names,
                **kwargs)
            for name, reg in internal_register_dict.items():
                if reg.is_output:
                    # Extending output register when register is repeated measurement
                    # bit register
                    if (internal_register_dict[name].vartype == 'bit'
                            and hasattr(reg.register[0], '__iter__')):
                        output_register_dict[name].extend(reg)
                    else:
                        output_register_dict[name].append(reg)
            # Overwriting global phase
            if 'global_phase' in output_register_dict.keys() and global_phase != 0:
                output_register_dict['global_phase'].register = [[global_phase]]

            if keep_state and self.qureg.isDensityMatrix:
                self.state = pyquest_cffi.cheat.getDensityMatrix()(qureg=self.qureg)
            elif keep_state:
                self.state = pyquest_cffi.cheat.getStateVector()(qureg=self.qureg)
            pyquest_cffi.utils.destroyQureg()(env=env, qubits=self.qureg)
            del self.qureg
        pyquest_cffi.utils.destroyQuestEnv()(env)

        return output_register_dict
