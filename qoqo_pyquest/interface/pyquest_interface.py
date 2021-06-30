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
"""Define the PyQuEST interface for qoqo operations and circuits."""

from qoqo import operations as ops
from qoqo import Circuit
from pyquest_cffi import ops as qops
from pyquest_cffi import cheat as qcheat
from pyquest_cffi import utils as qutils
from pyquest_cffi.questlib import (
    _PYQUEST,
    tqureg
)
from typing import (
    Dict,
    Any,
    Union,
    cast,
    List,
    Sequence,
)
from qoqo_calculator_pyo3 import (
    CalculatorFloat
)
import numpy as np

# Create look-up tables

_ALLOWED_PRAGMAS = ['PragmaSetNumberOfMeasurements',
                    'PragmaSetStateVector',
                    'PragmaDamping',
                    'PragmaDepolarise',
                    'PragmaDephasing',
                    'PragmaRandomNoise',
                    'PragmaRepeatGate',
                    'PragmaBoostNoise',
                    'PragmaOverrotation',
                    'PragmaStop',
                    'PragmaGlobalPhase',
                    'PragmaOverrotation']


# Defining the actual call

def pyquest_call_circuit(
        circuit: Circuit,
        qureg: tqureg,
        classical_bit_registers: Dict[str, List[bool]],
        classical_float_registers: Dict[str, List[float]],
        classical_complex_registers: Dict[str, List[complex]],
        output_bit_register_dict: Dict[str, List[List[bool]]],
) -> None:
    """Execute the qoqo circuit with PyQuEST

    The PyQuEST package is used to simulate a quantum computer, and this interface provides the
    translation of a qoqo circuit (or just a qoqo operation) to PyQuEST.
    The results are stored in the classical registers (dictionaries of registers).

    Args:
        circuit: The qoqo circuit that is executed
        qureg: The quantum register pyquest_cffi operates on
        classical_bit_registers: Dictionary or registers (lists) containing bit readout values
        classical_float_registers: Dictionary or registers (lists)
                                   containing float readout values
        classical_complex_registers: Dictionary or registers (lists)
                                     containing complex readout values
        output_bit_register_dict: Dictionary or lists of registers (lists)
                              containing a register for each repetition of the circuit
    """
    for op in circuit:
        pyquest_call_operation(op,
                               qureg,
                               classical_bit_registers,
                               classical_float_registers,
                               classical_complex_registers,
                               output_bit_register_dict,
                               )


def pyquest_call_operation(
        operation: Any,
        qureg: tqureg,
        classical_bit_registers: Dict[str, List[bool]],
        classical_float_registers: Dict[str, List[float]],
        classical_complex_registers: Dict[str, List[complex]],
        output_bit_register_dict: Dict[str, List[List[bool]]],
) -> None:
    """Execute the qoqo operation with PyQuEST

    Args:
        operation: The qoqo operation that is executed
        qureg: The quantum register pyquest_cffi operates on
        classical_bit_registers: Dictionary or registers (lists) containing bit readout values
        classical_float_registers: Dictionary or registers (lists) containing float readout values
        classical_complex_registers: Dictionary or registers (lists)
                                     containing complex readout values
        output_bit_register_dict: Dictionary or lists of registers (lists)
                              containing a register for each repetition of the circuit

    Raises:
        RuntimeError: Operation not in PyQuEST backend
    """
    tags = operation.tags()
    if 'RotateZ' in tags:
        operation = cast(ops.RotateZ, operation)
        qops.rotateZ()(qureg=qureg, qubit=operation.qubit(), theta=operation.theta().float())
    elif 'RotateX' in tags:
        operation = cast(ops.RotateX, operation)
        qops.rotateX()(qureg=qureg, qubit=operation.qubit(), theta=operation.theta().float())
    elif 'RotateY' in tags:
        operation = cast(ops.RotateY, operation)
        qops.rotateY()(qureg=qureg, qubit=operation.qubit(), theta=operation.theta().float())
    elif 'SqrtISwap' in tags:
        operation = cast(ops.SqrtISwap, operation)
        qops.sqrtISwap()(qureg=qureg, control=operation.control(), qubit=operation.target())
    elif 'CNOT' in tags:
        operation = cast(ops.CNOT, operation)
        qops.controlledNot()(qureg=qureg, control=operation.control(), qubit=operation.target())
    elif 'InvSqrtISwap' in tags:
        operation = cast(ops.InvSqrtISwap, operation)
        qops.invSqrtISwap()(qureg=qureg, control=operation.control(), qubit=operation.target())
    elif 'Hadamard' in tags:
        operation = cast(ops.Hadamard, operation)
        qops.hadamard()(qureg=qureg, qubit=operation.qubit())
    elif 'PauliX' in tags:
        operation = cast(ops.PauliX, operation)
        qops.pauliX()(qureg=qureg, qubit=operation.qubit())
    elif 'PauliY' in tags:
        operation = cast(ops.PauliY, operation)
        qops.pauliY()(qureg=qureg, qubit=operation.qubit())
    elif 'PauliZ' in tags:
        operation = cast(ops.PauliZ, operation)
        qops.pauliZ()(qureg=qureg, qubit=operation.qubit())
    elif 'SGate' in tags:
        operation = cast(ops.SGate, operation)
        qops.sGate()(qureg=qureg, qubit=operation.qubit())
    elif 'TGate' in tags:
        operation = cast(ops.TGate, operation)
        qops.tGate()(qureg=qureg, qubit=operation.qubit())
    elif 'SqrtPauliX' in tags:
        qops.rotateX()(qureg=qureg, qubit=operation.qubit(), theta=np.pi / 2)
    elif 'InvSqrtPauliX' in tags:
        qops.rotateX()(qureg=qureg, qubit=operation.qubit(), theta=-np.pi / 2)
    elif 'ControlledPhaseShift' in tags:
        qops.controlledPhaseShift()(qureg=qureg, control=operation.control(),
                                    qubit=operation.target(), theta=operation.theta().float())
    elif 'ControlledPauliY' in tags:
        qops.controlledPauliY()(qureg=qureg, control=operation.control(),
                                qubit=operation.target())
    elif 'ControlledPauliZ' in tags:
        qops.controlledPhaseFlip()(qureg=qureg, control=operation.control(),
                                   qubit=operation.target())
    elif 'RotateAroundSphericalAxis' in tags:
        qops.rotateAroundSphericalAxis()(
            qureg=qureg, qubit=operation.qubit(), theta=operation.theta().float(),
            spherical_theta=operation.spherical_theta().float(),
            spherical_phi=operation.spherical_phi().float())
    elif 'SingleQubitGateOperation' in tags:
        qops.compactUnitary()(qubit=operation.qubit(),
                              qureg=qureg,
                              alpha=operation.alpha_r().float() + 1j * operation.alpha_i().float(),
                              beta=operation.beta_r().float() + 1j * operation.beta_i().float())
    elif 'TwoQubitGateOperation' in tags:
        qops.twoQubitUnitary()(qureg=qureg, target_qubit_2=operation.control(),
                               target_qubit_1=operation.target(), matrix=operation.unitary_matrix())
    elif 'PragmaRepeatedMeasurement' in tags:
        _execute_PragmaRepeatedMeasurement(
            operation, qureg,
            classical_bit_registers,
            output_bit_register_dict,
        )
    elif 'PragmaDamping' in tags:
        qops.mixDamping()(
            qureg=qureg, qubit=operation.qubit(), probability=operation.probability().float())
    elif 'PragmaDepolarising' in tags:
        qops.mixDepolarising()(
            qureg=qureg, qubit=operation.qubit(), probability=operation.probability().float())
    elif 'PragmaDephasing' in tags:
        qops.mixDephasing()(
            qureg=qureg, qubit=operation.qubit(), probability=operation.probability().float())
    elif 'PragmaSetStateVector' in tags:
        vector = operation.statevector()
        qcheat.initStateFromAmps()(qureg=qureg, reals=np.real(vector), imags=np.imag(vector))
    elif 'PragmaSetDensityMatrix' in tags:
        density_matrix = operation.density_matrix()
        dim = int(np.round(np.sqrt(density_matrix.shape[0])))
        density_matrix = density_matrix.reshape((dim, dim))
        qcheat.initStateFromAmps()(
            qureg=qureg, reals=np.real(density_matrix), imags=np.imag(density_matrix))
    elif 'PragmaRandomNoise' in tags:
        _execute_PragmaRandomNoise(
            operation, qureg,
        )
    elif 'PragmaActiveReset' in tags:
        _execute_PragmaActiveReset(
            operation, qureg,
        )
    elif 'MeasureQubit' in tags:
        _execute_MeasureQubit(
            operation, qureg, classical_bit_registers,
        )
    elif 'InputDefinition' in tags:
        if operation.name() in classical_float_registers.keys():
            classical_float_registers = operation.input()
    elif 'Definition' in tags:
        pass
    elif 'PragmaGetPauliProduct' in tags:
        _execute_GetPauliProduct(
            operation, qureg,
            classical_float_registers,
        )
    elif 'PragmaGetStateVector' in tags:
        _execute_GetStateVector(
            operation, qureg,
            classical_complex_registers,
        )
    elif 'PragmaGetDensityMatrix' in tags:
        _execute_GetStateVector(
            operation, qureg,
            classical_complex_registers,
        )
    elif 'PragmaGetOccupationProbability' in tags:
        _execute_GetOccupationProbability(
            operation, qureg,
            classical_float_registers,
        )
    elif 'PragmaGetRotatedOccupationProbability' in tags:
        _execute_GetOccupationProbability(
            operation, qureg,
            classical_float_registers,
        )
    elif 'PragmaConditional' in tags:
        cast(ops.PragmaConditional, operation)
        if operation.condition_register() not in classical_bit_registers.keys():
            raise RuntimeError(
                "Conditional register {} not found in classical bit registers".format(
                    operation.condition_register()))
        if classical_bit_registers[operation.condition_register()][operation.condition_index()]:
            pyquest_call_circuit(
                circuit=operation.circuit(),
                qureg=qureg,
                classical_bit_registers=classical_bit_registers,
                classical_float_registers=classical_float_registers,
                classical_complex_registers=classical_complex_registers,
                output_bit_register_dict=output_bit_register_dict,
            )
    elif any(pragma in tags for pragma in _ALLOWED_PRAGMAS):
        pass
    else:
        raise RuntimeError('Operation not in PyQuEST backend')


def _execute_GetPauliProduct(
        operation: ops.PragmaGetPauliProduct,
        qureg: tqureg,
        classical_float_registers: Dict[str, List[float]],
) -> None:

    if operation.qubit_paulis == {}:
        classical_float_registers[operation.readout()] = [1.0, ]
        return None
    N = qureg.numQubitsRepresented
    env = qutils.createQuestEnv()()
    if qureg.isDensityMatrix:
        workspace = qutils.createDensityQureg()(N, env)
        workspace_pp = qutils.createDensityQureg()(N, env)
    else:
        workspace = qutils.createQureg()(N, env)
        workspace_pp = qutils.createQureg()(N, env)
    qutils.cloneQureg()(workspace, qureg)
    if operation.circuit() is not None:
        pyquest_call_circuit(
            circuit=operation.circuit(),
            qureg=workspace,
            classical_bit_registers=dict(),
            classical_float_registers=dict(),
            classical_complex_registers=dict(),
            output_bit_register_dict=dict(),
        )
    qubits = list()
    paulis = list()
    for qubit, pauli in operation.qubit_paulis().items():
        qubits.append(qubit)
        paulis.append(pauli)
    pp = qcheat.calcExpecPauliProd().call_interactive(
        qureg=workspace, qubits=qubits, paulis=paulis, workspace=workspace_pp)
    qutils.destroyQureg()(qubits=workspace_pp, env=env)
    qutils.destroyQureg()(qubits=workspace, env=env)
    del workspace
    qutils.destroyQuestEnv()(env)
    del env
    classical_float_registers[operation.readout()] = [pp, ]


def _execute_GetStateVector(
        operation: Union[ops.PragmaGetStateVector, ops.PragmaGetDensityMatrix],
        qureg: tqureg,
        classical_complex_registers: Dict[str, List[complex]],
) -> None:
    tags = operation.tags()
    if 'PragmaGetStateVector' in tags:
        is_state = True
        quest_obj = qcheat.getStateVector()
    if 'PragmaGetDensityMatrix' in tags:
        is_state = False
        quest_obj = qcheat.getDensityMatrix()

    if operation.circuit() is None:
        if is_state is True:
            statevector = quest_obj(qureg=qureg, )
        else:
            density_matrix = quest_obj(qureg=qureg, )
    else:
        N = qureg.numQubitsRepresented
        env = qutils.createQuestEnv()()
        if qureg.isDensityMatrix:
            workspace = qutils.createDensityQureg()(N, env)
        else:
            workspace = qutils.createQureg()(N, env)
        qutils.cloneQureg()(workspace, qureg)
        if operation.circuit() is not None:
            pyquest_call_circuit(
                circuit=operation.circuit(),
                qureg=workspace,
                classical_bit_registers=dict(),
                classical_float_registers=dict(),
                classical_complex_registers=dict(),
                output_bit_register_dict=dict(),)
        if is_state is True:
            statevector = quest_obj(qureg=workspace)
        else:
            density_matrix = quest_obj(qureg=workspace).flatten()

    classical_complex_registers[
        operation.readout()] = statevector if is_state is True else density_matrix


def _execute_GetOccupationProbability(
        operation: Union[
            ops.PragmaGetOccupationProbability],
        qureg: tqureg,
        classical_float_registers: Dict[str, List[float]],
) -> None:
    tags = operation.tags()
    quest_obj = qcheat.getOccupationProbability()
    if 'PragmaGetRotatedOccupationProbability' in tags:
        N = qureg.numQubitsRepresented
        env = qutils.createQuestEnv()()
        if qureg.isDensityMatrix:
            workspace = qutils.createDensityQureg()(N, env)
        else:
            workspace = qutils.createQureg()(N, env)
        qutils.cloneQureg()(workspace, qureg)
        if operation.circuit() is not None:
            pyquest_call_circuit(
                circuit=operation.circuit(),
                qureg=workspace,
                classical_bit_registers=dict(),
                classical_float_registers=dict(),
                classical_complex_registers=dict(),
                output_bit_register_dict=dict(),)
        occ_probabilities = np.real(quest_obj(qureg=workspace))
        qutils.destroyQureg()(qubits=workspace, env=env)
        del workspace
        qutils.destroyQuestEnv()(env)
        del env
    else:
        occ_probabilities = np.real(quest_obj(qureg=qureg, ))

    classical_float_registers[operation.readout()] = occ_probabilities


def _execute_PragmaRepeatedMeasurement(
        operation: ops.PragmaRepeatedMeasurement,
        qureg: tqureg,
        classical_bit_registers: Dict[str, List[bool]],
        output_bit_register_dict: Dict[str, List[List[bool]]],
) -> None:
    index_dict = operation.qubit_mapping()
    N = qureg.numQubitsRepresented
    if index_dict is None:
        index_dict = {i: i for i in range(N)}
    meas = qcheat.getRepeatedMeasurement()(
        qureg=qureg,
        number_measurements=operation.number_measurements(),
        qubits_to_readout_index_dict=index_dict)
    del classical_bit_registers[operation.readout()]
    output_bit_register_dict[operation.readout()] = meas.tolist()


def _execute_PragmaRandomNoise(
        operation: ops.PragmaRandomNoise,
        qureg: tqureg,
) -> None:
    applied_pauli = 0
    r0 = np.random.rand()
    rates = [(operation.depolarising_rate() / 4),
             (operation.depolarising_rate() / 4),
             ((operation.depolarising_rate() / 4)
              + (operation.dephasing_rate()))]
    rrates: List[float] = list()

    for r in rates:
        rrates.append(r.float())
    rates = rrates
    gate_time = float(operation.gate_time())
    rates = cast(List[CalculatorFloat], rates)
    # t0 = -np.log(r0) / np.sum(rates)
    probabilities = np.zeros((3,))
    for co, gamma in enumerate(rates):
        probabilities[co] = 0 if CalculatorFloat(gamma).isclose(0) else gamma
    if np.sum(probabilities) != 0:
        probabilities_normalised = probabilities / np.sum(probabilities)
    # max_iteration_counter = 0
    if r0 < 1 - gate_time * np.sum(rates):
        applied_pauli = 0
    else:
        applied_pauli = int(np.random.choice([1, 2, 3],
                                             p=probabilities_normalised))
    if applied_pauli == 0:
        return None
    else:
        quest_objects = [None,
                         qops.pauliX(),
                         qops.pauliY(),
                         qops.pauliZ()]
        quest_obj = cast(_PYQUEST, quest_objects[int(applied_pauli)])
        quest_obj(qureg=qureg, qubit=operation.qubit())


def _execute_PragmaActiveReset(
        operation: ops.PragmaActiveReset,
        qureg: tqureg,
) -> None:
    qubit = operation.qubit()
    res = qops.measure().call_interactive(qureg=qureg, qubit=qubit)
    if res == 1:
        qops.pauliX().call_interactive(qureg=qureg, qubit=qubit)


def _execute_MeasureQubit(
        operation: ops.MeasureQubit,
        qureg: tqureg,
        classical_bit_registers: Dict[str, List[bool]],
) -> None:
    res = qops.measure().call_interactive(qureg=qureg, qubit=operation.qubit())
    classical_bit_registers[operation.readout()][operation.readout_index()] = bool(res)


def basis_state_to_index(basis_state: Union[np.ndarray, Sequence[float]],
                         qubit_mapping: Dict[int, int]) -> int:
    """Convert basis state to index.

    Converts an up/down representation of a basis state to the index of the basis
    depending on the Endian convention of the system

    Args:
        basis_state: a sequence of 0 and one representing the qubit basis state
        qubit_mapping: Mapping of qubits to indeces in readout register

    Returns:
        int

    """
    b_state = np.array(basis_state)
    index = np.sum(np.dot(np.array(
        [2**qubit_mapping[k] for k in range(0, len(b_state))]), b_state))
    return index


def index_to_basis_state(index: int, number_qubits: int) -> List[int]:
    """Convert index to basis state.

    Converts an index of the basis to the up/down representation of a basis state
    depending on the Endian convention of the system

    Args:
        index: the basis index
        number_qubits: The number of qubits

    Returns:
        List[ingt]: A list of 0 and one representing the qubit basis state

    """
    b_list = list()
    for k in range(0, number_qubits):
        b_list.append((index // 2**k) % 2)
    basis_state = b_list
    return basis_state
