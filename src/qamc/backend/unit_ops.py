"""Low-level matrix operations used by density-matrix backends."""

from __future__ import annotations

from math import cos, sin, sqrt

import numpy as np


def int_to_bits(index: int, width: int) -> tuple[int, ...]:
    return tuple((index >> (width - 1 - offset)) & 1 for offset in range(width))


def bits_to_int(bits: tuple[int, ...] | list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def zero_density_matrix(num_qubits: int) -> np.ndarray:
    dim = 2**num_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def zero_state_vector(num_qubits: int) -> np.ndarray:
    dim = 2**num_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    return state


def state_vector_to_density_matrix(state: np.ndarray) -> np.ndarray:
    vector = np.asarray(state, dtype=complex).reshape(-1)
    return np.outer(vector, vector.conj())


def named_gate_matrix(name: str, params: tuple[object, ...] = ()) -> np.ndarray:
    gate_name = name.upper()
    if gate_name == "ID":
        return np.eye(2, dtype=complex)
    if gate_name == "H":
        return np.array([[1, 1], [1, -1]], dtype=complex) / sqrt(2)
    if gate_name == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if gate_name == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if gate_name == "P":
        theta = _single_gate_angle(name, params)
        return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)
    if gate_name == "CX":
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=complex,
        )
    if gate_name == "CP":
        theta = _single_gate_angle(name, params)
        return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(complex)
    if gate_name == "RBS":
        theta = _single_gate_angle(name, params)
        c = cos(theta)
        s = sin(theta)
        return np.array(
            [
                [1, 0, 0, 0],
                [0, c, s, 0],
                [0, -s, c, 0],
                [0, 0, 0, 1],
            ],
            dtype=complex,
        )
    if gate_name == "CCX":
        matrix = np.eye(8, dtype=complex)
        matrix[6, 6] = 0
        matrix[7, 7] = 0
        matrix[6, 7] = 1
        matrix[7, 6] = 1
        return matrix
    raise ValueError(f"Unsupported gate: {name!r}")


def _single_gate_angle(name: str, params: tuple[object, ...]) -> float:
    if len(params) != 1:
        raise ValueError(f"Gate {name!r} expects one angle parameter.")
    return float(params[0])


def expand_local_operator(
    local_operator: np.ndarray,
    qubit_order: tuple[str, ...],
    acted_qubits: tuple[str, ...],
) -> np.ndarray:
    n = len(qubit_order)
    k = len(acted_qubits)
    if k == 0:
        return np.eye(2**n, dtype=complex)

    positions = tuple(qubit_order.index(qubit) for qubit in acted_qubits)
    dim = 2**n
    full = np.zeros((dim, dim), dtype=complex)

    for input_index in range(dim):
        input_bits = list(int_to_bits(input_index, n))
        local_input_bits = tuple(input_bits[position] for position in positions)
        local_input_index = bits_to_int(local_input_bits)
        for local_output_index in range(2**k):
            amplitude = local_operator[local_output_index, local_input_index]
            if abs(amplitude) == 0:
                continue
            output_bits = list(input_bits)
            local_output_bits = int_to_bits(local_output_index, k)
            for position, bit in zip(positions, local_output_bits):
                output_bits[position] = bit
            output_index = bits_to_int(output_bits)
            full[output_index, input_index] += amplitude
    return full


def apply_local_operator(
    rho: np.ndarray,
    qubit_order: tuple[str, ...],
    local_operator: np.ndarray,
    acted_qubits: tuple[str, ...],
) -> np.ndarray:
    full_operator = expand_local_operator(local_operator, qubit_order, acted_qubits)
    return full_operator @ rho @ full_operator.conj().T


def apply_local_operator_to_state_vector(
    state: np.ndarray,
    qubit_order: tuple[str, ...],
    local_operator: np.ndarray,
    acted_qubits: tuple[str, ...],
) -> np.ndarray:
    n = len(qubit_order)
    k = len(acted_qubits)
    if k == 0:
        return np.asarray(state, dtype=complex).copy()

    positions = tuple(qubit_order.index(qubit) for qubit in acted_qubits)
    rest_positions = tuple(position for position in range(n) if position not in positions)
    permutation = positions + rest_positions
    inverse_permutation = np.argsort(permutation)

    tensor = np.asarray(state, dtype=complex).reshape((2,) * n)
    front = np.transpose(tensor, permutation).reshape(2**k, 2 ** (n - k))
    local = np.asarray(local_operator, dtype=complex).reshape(2**k, 2**k)
    updated = local @ front
    restored = np.transpose(updated.reshape((2,) * n), inverse_permutation)
    return restored.reshape(2**n)


def partial_trace(
    rho: np.ndarray,
    qubit_order: tuple[str, ...],
    retained_qubits: tuple[str, ...],
) -> np.ndarray:
    if retained_qubits == qubit_order:
        return rho.copy()

    n = len(qubit_order)
    retain_positions = tuple(qubit_order.index(qubit) for qubit in retained_qubits)
    trace_positions = tuple(pos for pos in range(n) if pos not in retain_positions)
    permutation = retain_positions + trace_positions

    tensor = rho.reshape((2,) * (2 * n))
    tensor = np.transpose(tensor, permutation + tuple(n + pos for pos in permutation))

    current_n = n
    retained_n = len(retained_qubits)
    for position in range(n - 1, retained_n - 1, -1):
        tensor = np.trace(tensor, axis1=position, axis2=current_n + position)
        current_n -= 1

    return tensor.reshape((2**retained_n, 2**retained_n))


def state_vector_partial_trace(
    state: np.ndarray,
    qubit_order: tuple[str, ...],
    retained_qubits: tuple[str, ...],
) -> np.ndarray:
    n = len(qubit_order)
    retained_n = len(retained_qubits)
    retain_positions = tuple(qubit_order.index(qubit) for qubit in retained_qubits)
    trace_positions = tuple(pos for pos in range(n) if pos not in retain_positions)
    permutation = retain_positions + trace_positions

    tensor = np.asarray(state, dtype=complex).reshape((2,) * n)
    reordered = np.transpose(tensor, permutation).reshape(2**retained_n, 2 ** (n - retained_n))
    return reordered @ reordered.conj().T


def reorder_density_matrix(
    rho: np.ndarray,
    current_order: tuple[str, ...],
    desired_order: tuple[str, ...],
) -> np.ndarray:
    if current_order == desired_order:
        return rho.copy()
    if set(current_order) != set(desired_order):
        raise ValueError("Current and desired qubit orders must contain the same qubits.")

    n = len(current_order)
    permutation = tuple(current_order.index(qubit) for qubit in desired_order)
    tensor = rho.reshape((2,) * (2 * n))
    tensor = np.transpose(tensor, permutation + tuple(n + pos for pos in permutation))
    return tensor.reshape((2**n, 2**n))


def reorder_state_vector(
    state: np.ndarray,
    current_order: tuple[str, ...],
    desired_order: tuple[str, ...],
) -> np.ndarray:
    if current_order == desired_order:
        return np.asarray(state, dtype=complex).copy()
    if set(current_order) != set(desired_order):
        raise ValueError("Current and desired qubit orders must contain the same qubits.")

    n = len(current_order)
    permutation = tuple(current_order.index(qubit) for qubit in desired_order)
    tensor = np.asarray(state, dtype=complex).reshape((2,) * n)
    return np.transpose(tensor, permutation).reshape(2**n)


def state_vector_distribution(
    state: np.ndarray,
    qubit_order: tuple[str, ...],
    observed_qubits: tuple[str, ...],
) -> dict[object, float]:
    n = len(qubit_order)
    k = len(observed_qubits)
    positions = tuple(qubit_order.index(qubit) for qubit in observed_qubits)
    rest_positions = tuple(position for position in range(n) if position not in positions)
    permutation = positions + rest_positions

    probabilities = np.abs(np.asarray(state, dtype=complex).reshape((2,) * n)) ** 2
    marginal = np.transpose(probabilities, permutation).reshape(2**k, 2 ** (n - k)).sum(axis=1)

    distribution: dict[object, float] = {}
    for index, probability in enumerate(marginal):
        bits = int_to_bits(index, k)
        key: object = bits[0] if len(bits) == 1 else bits
        distribution[key] = float(probability)
    return distribution


def single_qubit_projectors(basis: str) -> tuple[np.ndarray, np.ndarray]:
    basis_name = basis.upper()
    if basis_name == "Z":
        ket0 = np.array([[1.0], [0.0]], dtype=complex)
        ket1 = np.array([[0.0], [1.0]], dtype=complex)
        return ket0 @ ket0.conj().T, ket1 @ ket1.conj().T
    if basis_name == "X":
        ket_plus = np.array([[1.0], [1.0]], dtype=complex) / sqrt(2)
        ket_minus = np.array([[1.0], [-1.0]], dtype=complex) / sqrt(2)
        return ket_plus @ ket_plus.conj().T, ket_minus @ ket_minus.conj().T
    raise ValueError(f"Unsupported basis: {basis!r}")
