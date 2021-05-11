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
    Optional,
    Dict,
    Union,
    cast,
    List,
    Sequence,
    Tuple
)
from hqsbase.calculator import (
    Calculator,
    CalculatorFloat
)
from qoqo.registers import (
    BitRegister,
    FloatRegister,
    ComplexRegister,
)
import numpy as np
from scipy import sparse as sp


# Create look-up tables

_PYQUEST_ARGUMENT_NAME_DICTS: Dict[str, Dict[str, CalculatorFloat]] = dict()
_ADDITIONAL_QUEST_ARGS: Dict[str, Dict[str, float]] = dict()
_QUEST_OBJECTS: Dict[str, _PYQUEST] = dict()

_PYQUEST_ARGUMENT_NAME_DICTS['RotateZ'] = {'qubit': ('qubits', 'qubit'),
                                           'theta': ('parameters', 'theta')}
_QUEST_OBJECTS['RotateZ'] = qops.rotateZ()
_PYQUEST_ARGUMENT_NAME_DICTS['RotateX'] = {'qubit': ('qubits', 'qubit'),
                                           'theta': ('parameters', 'theta')}
_QUEST_OBJECTS['RotateX'] = qops.rotateX()
_PYQUEST_ARGUMENT_NAME_DICTS['RotateY'] = {'qubit': ('qubits', 'qubit'),
                                           'theta': ('parameters', 'theta')}
_QUEST_OBJECTS['RotateY'] = qops.rotateY()
_PYQUEST_ARGUMENT_NAME_DICTS['SingleQubitGate'] = {'qubit': ('qubits', 'qubit'),
                                                   'alpha': ('parameters', [('alpha_r', 1),
                                                                            ('alpha_i', 1j)]),
                                                   'beta': ('parameters', [('beta_r', 1),
                                                                           ('beta_i', 1j)])}
_QUEST_OBJECTS['SingleQubitGate'] = qops.compactUnitary()
_PYQUEST_ARGUMENT_NAME_DICTS['Hadamard'] = {'qubit': ('qubits', 'qubit')}
_QUEST_OBJECTS['Hadamard'] = qops.hadamard()
_PYQUEST_ARGUMENT_NAME_DICTS['PauliX'] = {'qubit': ('qubits', 'qubit')}
_QUEST_OBJECTS['PauliX'] = qops.pauliX()
_PYQUEST_ARGUMENT_NAME_DICTS['PauliY'] = {'qubit': ('qubits', 'qubit')}
_QUEST_OBJECTS['PauliY'] = qops.pauliY()
_PYQUEST_ARGUMENT_NAME_DICTS['PauliZ'] = {'qubit': ('qubits', 'qubit')}
_QUEST_OBJECTS['PauliZ'] = qops.pauliZ()
_PYQUEST_ARGUMENT_NAME_DICTS['SGate'] = {'qubit': ('qubits', 'qubit')}
_QUEST_OBJECTS['SGate'] = qops.sGate()
_PYQUEST_ARGUMENT_NAME_DICTS['TGate'] = {'qubit': ('qubits', 'qubit')}
_QUEST_OBJECTS['TGate'] = qops.tGate()
_PYQUEST_ARGUMENT_NAME_DICTS['SqrtPauliX'] = {'qubit': ('qubits', 'qubit')}
_ADDITIONAL_QUEST_ARGS['SqrtPauliX'] = {'theta': np.pi / 2}
_QUEST_OBJECTS['SqrtPauliX'] = qops.rotateX()
_PYQUEST_ARGUMENT_NAME_DICTS['InvSqrtPauliX'] = {'qubit': ('qubits', 'qubit')}
_ADDITIONAL_QUEST_ARGS['InvSqrtPauliX'] = {'theta': -np.pi / 2}
_QUEST_OBJECTS['InvSqrtPauliX'] = qops.rotateX()
_PYQUEST_ARGUMENT_NAME_DICTS['RotateAroundSphericalAxis'] = {'qubit': ('qubits', 'qubit'),
                                                             'theta': ('parameters', 'theta'),
                                                             'spherical_phi': ('parameters',
                                                                               'spherical_phi'),
                                                             'spherical_theta': ('parameters',
                                                                                 'spherical_theta')}
_QUEST_OBJECTS['RotateAroundSphericalAxis'] = qops.rotateAroundSphericalAxis()
_PYQUEST_ARGUMENT_NAME_DICTS['W'] = {'qubit': ('qubits', 'qubit'),
                                     'theta': ('parameters', 'theta'),
                                     'spherical_phi': ('parameters', 'spherical_phi')}
_ADDITIONAL_QUEST_ARGS['W'] = {'spherical_theta': np.pi / 2}
_QUEST_OBJECTS['W'] = qops.rotateAroundSphericalAxis()
_PYQUEST_ARGUMENT_NAME_DICTS['CNOT'] = {'qubit': ('qubits', 'qubit'),
                                        'control': ('qubits', 'control')}
_QUEST_OBJECTS['CNOT'] = qops.controlledNot()
_PYQUEST_ARGUMENT_NAME_DICTS['SqrtISwap'] = {'qubit': ('qubits', 'qubit'),
                                             'control': ('qubits', 'control')}
_QUEST_OBJECTS['SqrtISwap'] = qops.sqrtISwap()
_PYQUEST_ARGUMENT_NAME_DICTS['InvSqrtISwap'] = {'qubit': ('qubits', 'qubit'),
                                                'control': ('qubits', 'control')}
_QUEST_OBJECTS['InvSqrtISwap'] = qops.invSqrtISwap()
_PYQUEST_ARGUMENT_NAME_DICTS['ControlledPhaseShift'] = {'qubit': ('qubits', 'qubit'),
                                                        'control': ('qubits', 'control'),
                                                        'theta': ('parameters', 'theta')}
_QUEST_OBJECTS['ControlledPhaseShift'] = qops.controlledPhaseShift()
_PYQUEST_ARGUMENT_NAME_DICTS['ControlledPauliY'] = {'qubit': ('qubits', 'qubit'),
                                                    'control': ('qubits', 'control')}
_QUEST_OBJECTS['ControlledPauliY'] = qops.controlledPauliY()
_PYQUEST_ARGUMENT_NAME_DICTS['ControlledPauliZ'] = {'qubit': ('qubits', 'qubit'),
                                                    'control': ('qubits', 'control')}
_QUEST_OBJECTS['ControlledPauliZ'] = qops.controlledPhaseFlip()
_PYQUEST_ARGUMENT_NAME_DICTS['MolmerSorensenXX'] = {'qubit': ('qubits', 'qubit'),
                                                    'control': ('qubits', 'control')}
_QUEST_OBJECTS['MolmerSorensenXX'] = qops.MolmerSorensenXX()

_PYQUEST_ARGUMENT_NAME_DICTS['PragmaSetStateVector'] = {}
_QUEST_OBJECTS['PragmaSetStateVector'] = qcheat.initStateFromAmps()
_PYQUEST_ARGUMENT_NAME_DICTS['PragmaSetDensityMatrix'] = {}
_QUEST_OBJECTS['PragmaSetDensityMatrix'] = qcheat.initStateFromAmps()
_PYQUEST_ARGUMENT_NAME_DICTS['PragmaDamping'] = {'qubit': ('qubits', 'qubit')}
_QUEST_OBJECTS['PragmaDamping'] = qops.mixDamping()
_PYQUEST_ARGUMENT_NAME_DICTS['PragmaDepolarise'] = {'qubit': ('qubits', 'qubit')}
_QUEST_OBJECTS['PragmaDepolarise'] = qops.mixDepolarising()
_PYQUEST_ARGUMENT_NAME_DICTS['PragmaDephasing'] = {'qubit': ('qubits', 'qubit')}
_QUEST_OBJECTS['PragmaDephasing'] = qops.mixDephasing()
_PYQUEST_ARGUMENT_NAME_DICTS['PragmaRandomNoise'] = {'qubit': ('qubits', 'qubit')}
_QUEST_OBJECTS['PragmaRandomNoise'] = qops.pauliZ()
_QUEST_OBJECTS['MeasureQubit'] = qops.measure()
_QUEST_OBJECTS['PragmaRepeatedMeasurement'] = qcheat.getRepeatedMeasurement()

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
                    'PragmaParameterSubstitution']


# Defining the actual call

def pyquest_call_circuit(
        circuit: Circuit,
        qureg: tqureg,
        classical_registers: Dict[str, Union[BitRegister,
                                             FloatRegister, ComplexRegister]],
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    """Execute the qoqo circuit with PyQuEST

    The PyQuEST package is used to simulate a quantum computer, and this interface provides the
    translation of a qoqo circuit (or just a qoqo operation) to PyQuEST.
    The results are stored in the classical registers (dictionaries of registers).

    Args:
        circuit: The qoqo circuit that is executed
        qureg: The quantum register pyquest_cffi operates on
        classical_registers: The classical registers used for input/output
        calculator: The HQSBase Calculator used to replace symbolic parameters
        qubit_names: The dictionary of qubit names to translate to
        **kwargs: Additional keyword arguments
    """
    for op in circuit:
        pyquest_call_operation(op,
                               qureg,
                               classical_registers,
                               calculator,
                               qubit_names,
                               **kwargs)


def pyquest_call_operation(
        operation: ops.Operation,
        qureg: tqureg,
        classical_registers: Dict[str, Union[BitRegister,
                                             FloatRegister, ComplexRegister]],
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    """Execute the qoqo operation with PyQuEST

    Args:
        operation: The qoqo operation that is executed
        qureg: The quantum register pyquest_cffi operates on
        classical_registers: The classical registers used for input/output
        calculator: The HQSBase Calculator used to replace symbolic parameters
        qubit_names: The dictionary of qubit names to translate to
        **kwargs: Additional keyword arguments

    Raises:
        OperationNotInBackendError: Operation not in PyQuEST backend
    """
    tags = operation._operation_tags
    if 'RotateZ' in tags:
        _execute_GateOperation(
            operation, qureg, 'RotateZ',
            calculator, qubit_names, **kwargs)
    elif 'RotateX' in tags:
        _execute_GateOperation(
            operation, qureg, 'RotateX',
            calculator, qubit_names, **kwargs)
    elif 'RotateY' in tags:
        _execute_GateOperation(
            operation, qureg, 'RotateY',
            calculator, qubit_names, **kwargs)
    elif 'SqrtISwap' in tags:
        _execute_GateOperation(
            operation, qureg, 'SqrtISwap',
            calculator, qubit_names, **kwargs)
    elif 'CNOT' in tags:
        _execute_GateOperation(
            operation, qureg, 'CNOT',
            calculator, qubit_names, **kwargs)
    elif 'InvSqrtISwap' in tags:
        _execute_GateOperation(
            operation, qureg, 'InvSqrtISwap',
            calculator, qubit_names, **kwargs)
    elif 'Hadamard' in tags:
        _execute_GateOperation(
            operation, qureg, 'Hadamard',
            calculator, qubit_names, **kwargs)
    elif 'PauliX' in tags:
        _execute_GateOperation(
            operation, qureg, 'PauliX',
            calculator, qubit_names, **kwargs)
    elif 'PauliY' in tags:
        _execute_GateOperation(
            operation, qureg, 'PauliY',
            calculator, qubit_names, **kwargs)
    elif 'PauliZ' in tags:
        _execute_GateOperation(
            operation, qureg, 'PauliZ',
            calculator, qubit_names, **kwargs)
    elif 'SGate' in tags:
        _execute_GateOperation(
            operation, qureg, 'SGate',
            calculator, qubit_names, **kwargs)
    elif 'TGate' in tags:
        _execute_GateOperation(
            operation, qureg, 'TGate',
            calculator, qubit_names, **kwargs)
    elif 'SqrtPauliX' in tags:
        _execute_GateOperation(
            operation, qureg, 'SqrtPauliX',
            calculator, qubit_names, **kwargs)
    elif 'InvSqrtPauliX' in tags:
        _execute_GateOperation(
            operation, qureg, 'InvSqrtPauliX',
            calculator, qubit_names, **kwargs)
    elif 'ControlledPhaseShift' in tags:
        _execute_GateOperation(
            operation, qureg, 'ControlledPhaseShift',
            calculator, qubit_names, **kwargs)
    elif 'ControlledPauliY' in tags:
        _execute_GateOperation(
            operation, qureg, 'ControlledPauliY',
            calculator, qubit_names, **kwargs)
    elif 'ControlledPauliZ' in tags:
        _execute_GateOperation(
            operation, qureg, 'ControlledPauliZ',
            calculator, qubit_names, **kwargs)
    elif 'RotateAroundSphericalAxis' in tags:
        _execute_GateOperation(
            operation, qureg, 'RotateAroundSphericalAxis',
            calculator, qubit_names, **kwargs)
    elif 'W' in tags:
        _execute_GateOperation(
            operation, qureg, 'W',
            calculator, qubit_names, **kwargs)
    # elif 'MolmerSorensenXX' in tags:
    #     _execute_GateOperation(
    #         operation, qureg, 'MolmerSorensenXX',
    #         calculator, qubit_names, **kwargs)
    elif 'SingleQubitGate' in tags:
        _execute_SingleQubitGate(
            operation, qureg,
            calculator, qubit_names, **kwargs)
    elif 'SingleQubitGateOperation' in tags:
        _execute_SingleQubitGateOperation(
            operation, qureg,
            calculator, qubit_names, **kwargs)
    elif 'TwoQubitGateOperation' in tags:
        _execute_TwoQubitGateOperation(
            operation, qureg,
            calculator, qubit_names, **kwargs)
    elif 'PragmaRepeatedMeasurement' in tags:
        _execute_PragmaRepeatedMeasurement(
            operation, qureg, classical_registers,
            calculator, qubit_names, **kwargs)
    elif 'PragmaDamping' in tags:
        _execute_GateOperation(
            operation, qureg, 'PragmaDamping',
            calculator, qubit_names, **kwargs)
    elif 'PragmaDepolarise' in tags:
        _execute_GateOperation(
            operation, qureg, 'PragmaDepolarise',
            calculator, qubit_names, **kwargs)
    elif 'PragmaDephasing' in tags:
        _execute_GateOperation(
            operation, qureg, 'PragmaDephasing',
            calculator, qubit_names, **kwargs)
    elif 'PragmaSetStateVector' in tags:
        _execute_GateOperation(
            operation, qureg, 'PragmaSetStateVector',
            calculator, qubit_names, **kwargs)
    elif 'PragmaSetDensityMatrix' in tags:
        _execute_GateOperation(
            operation, qureg, 'PragmaSetDensityMatrix',
            calculator, qubit_names, **kwargs)
    elif 'PragmaRandomNoise' in tags:
        _execute_PragmaRandomNoise(
            operation, qureg,
            calculator, qubit_names, **kwargs)
    elif 'PragmaActiveReset' in tags:
        _execute_PragmaActiveReset(
            operation, qureg,
            calculator, qubit_names, **kwargs)
    elif 'MeasureQubit' in tags:
        _execute_MeasureQubit(
            operation, qureg, classical_registers,
            calculator, qubit_names, **kwargs)
    elif 'Definition' in tags:
        pass
    elif 'PragmaGetPauliProduct' in tags:
        _execute_GetPauliProduct(
            operation, qureg, classical_registers,
            calculator, qubit_names)
    elif 'PragmaGetStateVector' in tags:
        _execute_GetStateVector(
            operation, qureg, classical_registers,
            calculator, qubit_names)
    elif 'PragmaGetDensityMatrix' in tags:
        _execute_GetStateVector(
            operation, qureg, classical_registers,
            calculator, qubit_names)
    elif 'PragmaGetOccupationProbability' in tags:
        _execute_GetOccupationProbability(
            operation, qureg, classical_registers,
            calculator, qubit_names)
    elif 'PragmaGetRotatedOccupationProbability' in tags:
        _execute_GetOccupationProbability(
            operation, qureg, classical_registers,
            calculator, qubit_names)
    elif 'PragmaPauliProdMeasurement' in tags:
        _execute_PragmaPauliProdMeasurement(
            operation, qureg, classical_registers,
            calculator, qubit_names)
    elif any(pragma in tags for pragma in _ALLOWED_PRAGMAS):
        pass
    else:
        raise ops.OperationNotInBackendError('Operation not in PyQuEST backend')


def _execute_GetPauliProduct(
        operation: ops.Operation,
        qureg: tqureg,
        classical_registers: Dict[str, Union[BitRegister,
                                             FloatRegister, ComplexRegister]],
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    operation = cast(ops.PragmaGetPauliProduct, operation)

    if np.isclose(np.sum(operation._pauli_product), 0):
        classical_registers[operation._readout].register = [1, ]
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
    if operation._circuit is not None:
        pyquest_call_circuit(
            circuit=operation._circuit,
            qureg=workspace,
            calculator=calculator,
            qubit_names=qubit_names,
            classical_registers=classical_registers)
    qubits = [i for i in range(N) if i in operation._pauli_product]
    paulis = [3 for _ in qubits]
    pp = qcheat.calcExpecPauliProd().call_interactive(
        qureg=workspace, qubits=qubits, paulis=paulis, workspace=workspace_pp)
    qutils.destroyQureg()(qubits=workspace_pp, env=env)
    qutils.destroyQureg()(qubits=workspace, env=env)
    del workspace
    qutils.destroyQuestEnv()(env)
    del env
    classical_registers[operation._readout].register = [pp, ]


def _execute_GetStateVector(
        operation: ops.Operation,
        qureg: tqureg,
        classical_registers: Dict[str, Union[BitRegister,
                                             FloatRegister, ComplexRegister]],
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    tags = operation._operation_tags
    set_operation: Union[ops.PragmaGetStateVector, ops.PragmaGetDensityMatrix]
    if 'PragmaGetStateVector' in tags:
        is_state = True
        set_operation = cast(ops.PragmaGetStateVector, operation)
        quest_obj = qcheat.getStateVector()
    if 'PragmaGetDensityMatrix' in tags:
        is_state = False
        set_operation = cast(ops.PragmaGetDensityMatrix, operation)
        quest_obj = qcheat.getDensityMatrix()

    if set_operation._circuit is None:
        if is_state is True:
            statevector = quest_obj(qureg=qureg, )
        else:
            density_matrix = quest_obj(qureg=qureg, )
        if set_operation._qubit_mapping is not None:
            max_mapping_qubit = np.max(list(set_operation._qubit_mapping.keys())) + 1
            data = list()
            ilist = list(range(2**max_mapping_qubit))
            jlist = list()
            for i in ilist:
                data.append(1)
                base = index_to_basis_state(i, number_qubits=max_mapping_qubit)
                jlist.append(basis_state_to_index(base, set_operation._qubit_mapping))
            if is_state is True:
                mapping_matrix = sp.coo_matrix(
                    (data, (jlist, ilist)),
                    shape=(len(statevector), 2**max_mapping_qubit))
                statevector = statevector @ mapping_matrix
            else:
                mapping_matrix = sp.coo_matrix(
                    (data, (jlist, ilist)),
                    shape=(density_matrix.shape[0], 2**max_mapping_qubit))
                density_matrix = mapping_matrix.T @ density_matrix @ mapping_matrix

        classical_registers[
            set_operation._readout].register = statevector if is_state is True else density_matrix
    else:
        N = qureg.numQubitsRepresented
        env = qutils.createQuestEnv()()
        if qureg.isDensityMatrix:
            workspace = qutils.createDensityQureg()(N, env)
        else:
            workspace = qutils.createQureg()(N, env)
        qutils.cloneQureg()(workspace, qureg)
        if set_operation._circuit is not None:
            pyquest_call_circuit(
                circuit=set_operation._circuit,
                qureg=workspace,
                calculator=calculator,
                qubit_names=qubit_names,
                classical_registers=classical_registers)
        if is_state is True:
            statevector = quest_obj(qureg=workspace)
        else:
            density_matrix = quest_obj(qureg=workspace)
        qutils.destroyQureg()(qubits=workspace, env=env)
        del workspace
        qutils.destroyQuestEnv()(env)
        del env
        if set_operation._qubit_mapping is not None:
            max_mapping_qubit = np.max(list(set_operation._qubit_mapping.keys())) + 1
            data = list()
            ilist = list(range(2**max_mapping_qubit))
            jlist = list()
            for i in ilist:
                data.append(1)
                base = index_to_basis_state(i, number_qubits=max_mapping_qubit)
                jlist.append(basis_state_to_index(base, set_operation._qubit_mapping))
            if is_state is True:
                mapping_matrix = sp.coo_matrix(
                    (data, (jlist, ilist)),
                    shape=(len(statevector), 2**max_mapping_qubit))
                statevector = statevector @ mapping_matrix
            else:
                mapping_matrix = sp.coo_matrix(
                    (data, (jlist, ilist)),
                    shape=(density_matrix.shape[0], 2**max_mapping_qubit))
                density_matrix = mapping_matrix.T @ density_matrix @ mapping_matrix
        classical_registers[
            set_operation._readout].register = statevector if is_state is True else density_matrix


def _execute_GetOccupationProbability(
        operation: ops.Operation,
        qureg: tqureg,
        classical_registers: Dict[str, Union[BitRegister,
                                             FloatRegister, ComplexRegister]],
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    operation = cast(ops.PragmaGetOccupationProbability, operation)
    tags = operation._operation_tags
    quest_obj = qcheat.getOccupationProbability()
    if 'PragmaGetRotatedOccupationProbability' in tags:
        operation = cast(ops.PragmaGetRotatedOccupationProbability, operation)
        N = qureg.numQubitsRepresented
        env = qutils.createQuestEnv()()
        if qureg.isDensityMatrix:
            workspace = qutils.createDensityQureg()(N, env)
        else:
            workspace = qutils.createQureg()(N, env)
        qutils.cloneQureg()(workspace, qureg)
        if operation._circuit is not None:
            pyquest_call_circuit(
                circuit=operation._circuit,
                qureg=workspace,
                calculator=calculator,
                qubit_names=qubit_names,
                classical_registers=classical_registers)
        occ_probabilities = np.real(quest_obj(qureg=workspace))
        qutils.destroyQureg()(qubits=workspace, env=env)
        del workspace
        qutils.destroyQuestEnv()(env)
        del env
    else:
        occ_probabilities = np.real(quest_obj(qureg=qureg, ))
        if operation._qubit_mapping is not None:
            max_mapping_qubit = np.max(list(operation._qubit_mapping.keys())) + 1
            data = list()
            ilist = list(range(2**max_mapping_qubit))
            jlist = list()
            for i in ilist:
                data.append(1)
                base = index_to_basis_state(i, number_qubits=max_mapping_qubit)
                jlist.append(basis_state_to_index(base, operation._qubit_mapping))
            mapping_matrix = sp.coo_matrix((data, (jlist, ilist)),
                                           shape=(len(occ_probabilities), 2**max_mapping_qubit))
            occ_probabilities = occ_probabilities @ mapping_matrix

    classical_registers[operation._readout].register = occ_probabilities


def _execute_PragmaPauliProdMeasurement(
        operation: ops.Operation,
        qureg: tqureg,
        classical_registers: Dict[str, Union[BitRegister,
                                             FloatRegister, ComplexRegister]],
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    operation = cast(ops.PragmaPauliProdMeasurement, operation)
    quest_obj = qcheat.calcExpecPauliProd()
    N = qureg.numQubitsRepresented
    env = qutils.createQuestEnv()()
    if qureg.isDensityMatrix:
        workspace = qutils.createDensityQureg()(N, env)
    else:
        workspace = qutils.createQureg()(N, env)

    meas = quest_obj(qureg=qureg, qubits=operation._qubits,
                     paulis=operation._paulis, workspace=workspace)
    qutils.destroyQureg(workspace)
    del workspace
    qutils.destroyQuestEnv(env)
    del env
    classical_registers[operation._readout].register[operation._readout_index] = meas


def _execute_SingleQubitGate(
        operation: ops.Operation,
        qureg: tqureg,
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    operation = cast(ops.SingleQubitGate, operation)
    quest_obj = cast(qops.compactUnitary, _QUEST_OBJECTS['SingleQubitGate'])
    quest_kwargs = dict()
    quest_kwargs['qureg'] = qureg
    if operation.is_parametrized and calculator is None:
        raise ops.OperationNotInBackendError(
            'Interactive PyQuEST can not be called with symbolic parameters'
            + ', substitute parameters first')
    parameter_dict: Dict[str, CalculatorFloat] = dict()
    if calculator is not None:
        for key, sarg in operation._ordered_parameter_dict.items():
            parameter_dict[key] = (calculator.parse_get(sarg.value))
    else:
        for key, sarg in operation._ordered_parameter_dict.items():
            parameter_dict[key] = sarg.value

    for key in _PYQUEST_ARGUMENT_NAME_DICTS['SingleQubitGate'].keys():
        dict_name, dict_key = _PYQUEST_ARGUMENT_NAME_DICTS['SingleQubitGate'][key]
        if dict_name == 'qubits':
            dict_key = cast(str, dict_key)
            arg = operation._ordered_qubits_dict[dict_key]
            if qubit_names is not None:
                arg = qubit_names[arg]
            quest_kwargs[key] = arg
        else:
            quest_kwargs[key] = 0
            dict_key = cast(List[Tuple[str, complex]], dict_key)
            for dkey, prefactor in dict_key:
                pval = parameter_dict[dkey]
                parg = pval * prefactor
                quest_kwargs[key] += parg
    quest_obj(**quest_kwargs)


def _execute_SingleQubitGateOperation(
        operation: ops.Operation,
        qureg: tqureg,
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    operation = cast(ops.SingleQubitGateOperation, operation)
    quest_obj = qops.compactUnitary()
    quest_kwargs = dict()
    quest_kwargs['qureg'] = qureg
    if operation.is_parametrized and calculator is None:
        raise ops.OperationNotInBackendError(
            'Interactive PyQuEST can not be called with symbolic parameters'
            + ', substitute parameters first')
    if calculator is not None:
        alpha_r = calculator.parse_get(operation.alpha_r.value)
        alpha_i = calculator.parse_get(operation.alpha_i.value)
        beta_r = calculator.parse_get(operation.beta_r.value)
        beta_i = calculator.parse_get(operation.beta_i.value)
    else:
        alpha_r = (operation.alpha_r.value)
        alpha_i = (operation.alpha_i.value)
        beta_r = (operation.beta_r.value)
        beta_i = (operation.beta_i.value)

    if qubit_names is not None:
        qubit = qubit_names[list(operation._ordered_qubits_dict.values())[0]]
    else:
        qubit = list(operation._ordered_qubits_dict.values())[0]
    quest_obj(qubit=qubit,
              qureg=qureg,
              alpha=alpha_r + 1j * alpha_i,
              beta=beta_r + 1j * beta_i)


def _execute_TwoQubitGateOperation(
        operation: ops.Operation,
        qureg: tqureg,
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    operation = cast(ops.TwoQubitGateOperation, operation)
    quest_obj = qops.twoQubitUnitary()
    if operation.is_parametrized and calculator is None:
        raise ops.OperationNotInBackendError(
            'Interactive PyQuEST can not be called with symbolic parameters'
            + ', substitute parameters first')
    parameter_dict: Dict[str, CalculatorFloat] = dict()
    if calculator is not None:
        for key, sarg in operation._ordered_parameter_dict.items():
            parameter_dict[key] = (calculator.parse_get(sarg.value))
    else:
        for key, sarg in operation._ordered_parameter_dict.items():
            parameter_dict[key] = sarg.value

    get_matrix = getattr(operation, 'unitary_matrix_from_parameters', None)
    if get_matrix is None:
        raise RuntimeError('Cannot find unitary matrix for gate')
    matrix = get_matrix(**parameter_dict)
    if ('control' in operation._ordered_qubits_dict.keys()
            and 'qubit' in operation._ordered_qubits_dict.keys()):
        control = operation._ordered_qubits_dict['control']
        qubit = operation._ordered_qubits_dict['qubit']
    elif('i' in operation._ordered_qubits_dict.keys()
            and 'j' in operation._ordered_qubits_dict.keys()):
        control = operation._ordered_qubits_dict['j']
        qubit = operation._ordered_qubits_dict['i']
    else:
        raise ValueError("Only 'control' and 'qubit' or 'i' and 'j' "
                         + "are allowed as qubit names in a two qubit gate")

    if qubit_names is not None:
        qubit = qubit_names[qubit]
        control = qubit_names[control]
    quest_obj(qureg=qureg,
              target_qubit_1=qubit,
              target_qubit_2=control,
              matrix=matrix)


def _execute_PragmaRepeatedMeasurement(
        operation: ops.Operation,
        qureg: tqureg,
        registers: Dict[str, Union[BitRegister,
                                   FloatRegister, ComplexRegister]],
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    operation = cast(ops.PragmaRepeatedMeasurement, operation)
    quest_obj = _QUEST_OBJECTS['PragmaRepeatedMeasurement']
    index_dict = operation._qubit_mapping
    N = qureg.numQubitsRepresented
    if index_dict is None:
        index_dict = {i: i for i in range(N)}
    meas = quest_obj(
        qureg=qureg,
        number_measurements=operation._number_measurements,
        qubits_to_readout_index_dict=index_dict)
    registers[operation._readout].register = meas


def _execute_PragmaRandomNoise(
        operation: ops.Operation,
        qureg: tqureg,
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    operation = cast(ops.PragmaRandomNoise, operation)
    quest_obj = _QUEST_OBJECTS['PragmaRandomNoise']
    if operation.is_parametrized and calculator is None:
        raise ops.OperationNotInBackendError(
            'Interactive PyQuEST can not be called with symbolic parameters'
            + ', substitute parameters first')
    applied_pauli = 0
    r0 = np.random.rand()
    rates = [(operation._ordered_parameter_dict['depolarisation_rate'] / 4),
             (operation._ordered_parameter_dict['depolarisation_rate'] / 4),
             ((operation._ordered_parameter_dict['depolarisation_rate'] / 4)
              + (operation._ordered_parameter_dict['dephasing_rate']))]
    rrates: List[float] = list()
    if calculator is not None:
        for r in rates:
            rrates.append((calculator.parse_get(r.value)))
        rates = rrates
        gate_time = (calculator.parse_get(
            operation._ordered_parameter_dict['gate_time'].value))
    else:
        for r in rates:
            rrates.append(r.value)
        rates = rrates
        gate_time = float(operation._ordered_parameter_dict['gate_time'])
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
        applied_pauli = np.random.choice([1, 2, 3],
                                         size=1,
                                         p=probabilities_normalised)
    if applied_pauli == 0:
        return None
    else:
        from pyquest_cffi import ops as qops
        quest_objects = [None,
                         qops.pauliX(),
                         qops.pauliY(),
                         qops.pauliZ()]
        quest_obj = cast(_PYQUEST, quest_objects[int(applied_pauli)])
        quest_kwargs: Dict[str, float] = dict()
        quest_kwargs['qureg'] = qureg

        for key in _PYQUEST_ARGUMENT_NAME_DICTS['PragmaRandomNoise'].keys():
            dict_name, dict_key = _PYQUEST_ARGUMENT_NAME_DICTS['PragmaRandomNoise'][key]
            if dict_name == 'qubits':
                arg = operation._ordered_qubits_dict[dict_key]
                if qubit_names is not None:
                    arg = qubit_names[arg]
                quest_kwargs[key] = arg
            else:
                parg = float(operation._ordered_parameter_dict[dict_key])
                quest_kwargs[key] = parg
        quest_obj(**quest_kwargs)


def _execute_PragmaActiveReset(
        operation: ops.Operation,
        qureg: tqureg,
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    operation = cast(ops.PragmaActiveReset, operation)
    if qubit_names is not None:
        qubit = qubit_names[operation._qubit]
    else:
        qubit = operation._qubit
    quest_obj_list = [qops.measure(), qops.pauliX()]
    res = quest_obj_list[0].call_interactive(qureg=qureg, qubit=qubit)
    if res == 1:
        quest_obj_list[1].call_interactive(qureg=qureg, qubit=qubit)


def _execute_MeasureQubit(
        operation: ops.Operation,
        qureg: tqureg,
        registers: Dict[str, Union[BitRegister,
                                   FloatRegister, ComplexRegister]],
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    operation = cast(ops.MeasureQubit, operation)
    if qubit_names is not None:
        qubit = qubit_names[operation._qubit]
    else:
        qubit = operation._qubit
    quest_obj = cast(qops.measure, _QUEST_OBJECTS['MeasureQubit'])
    res = quest_obj.call_interactive(qureg=qureg, qubit=qubit)
    registers[operation._readout].register[operation._readout_index] = res


def _execute_GateOperation(
        operation: ops.Operation,
        qureg: tqureg,
        tag: str,
        calculator: Optional[Calculator] = None,
        qubit_names: Optional[Dict[int, int]] = None,
        **kwargs) -> None:
    if tag in ['PragmaDamping', 'PragmaDepolarise', 'PragmaDephasing']:
        operation = cast(ops.PragmaNoise, operation)
        if calculator is not None:
            probability = (calculator.parse_get(operation.probability.value))
        else:
            probability = operation.probability.value
    elif tag in ['PragmaSetStateVector']:
        set_state = cast(ops.PragmaSetStateVector, operation)
    elif tag in ['PragmaSetDensityMatrix']:
        set_density = cast(ops.PragmaSetDensityMatrix, operation)
    else:
        operation = cast(ops.GateOperation, operation)
    quest_obj = _QUEST_OBJECTS[tag]
    quest_kwargs = dict()
    quest_kwargs['qureg'] = qureg
    if operation._parametrized is True and calculator is None:
        raise ops.OperationNotInBackendError(
            'Interactive PyQuEST can not be called with symbolic parameters'
            + ', substitute parameters first')

    parameter_dict: Dict[str, CalculatorFloat] = dict()
    if calculator is not None:
        for key, sarg in operation._ordered_parameter_dict.items():
            parameter_dict[key] = calculator.parse_get(sarg.value)
    else:
        for key, sarg in operation._ordered_parameter_dict.items():
            parameter_dict[key] = CalculatorFloat(sarg).value

    for key in _PYQUEST_ARGUMENT_NAME_DICTS[tag].keys():
        dict_name, dict_key = _PYQUEST_ARGUMENT_NAME_DICTS[tag][key]
        if dict_name == 'qubits':
            carg = operation._ordered_qubits_dict[dict_key]
            if qubit_names is not None:
                carg = qubit_names[carg]
            quest_kwargs[key] = carg
        else:
            parg = parameter_dict[dict_key]
            quest_kwargs[key] = parg
    additional_quest_args = _ADDITIONAL_QUEST_ARGS.get(tag, None)
    if additional_quest_args is not None:
        for key, val in additional_quest_args.items():
            quest_kwargs[key] = val

    if tag in ['PragmaDamping', 'PragmaDepolarise', 'PragmaDephasing']:
        quest_obj(probability=probability, **quest_kwargs)
    elif tag in ['PragmaSetStateVector']:
        quest_obj(qureg=qureg, reals=np.real(set_state._statevec),
                  imags=np.imag(set_state._statevec))
    elif tag in ['PragmaSetDensityMatrix']:
        quest_obj(qureg=qureg, reals=np.real(set_density._density_matrix),
                  imags=np.imag(set_density._density_matrix))
    else:
        quest_obj(**quest_kwargs)


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
