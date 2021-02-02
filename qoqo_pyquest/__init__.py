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
"""PyQuEST interface and backend for qoqo.

Translates qoqo operations and circuits to PyQuEST operations via the interface.
Calls PyQuest simulation via the backend and simulates the circuit.

.. autosummary::
    :toctree: generated/

    pyquest_call_operation
    pyquest_call_circuit
    PyQuestBackend

"""
from qoqo_pyquest.__version__ import __version__
from qoqo_pyquest.interface import (
    pyquest_call_operation,
    pyquest_call_circuit
)
from qoqo_pyquest.backend import PyQuestBackend
