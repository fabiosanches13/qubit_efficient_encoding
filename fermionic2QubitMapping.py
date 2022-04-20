from qiskit.chemistry import FermionicOperator
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli

import numpy as np
from math import ceil, log2
from copy import copy 

def num2binstr(n):
    return bin(n)[2:]

def lowbit(x):
    return x & (-x)

def generate_perms(n, until): 
    """
    (a math tool)
    Return a list of numbers with `n` ones in their binary expressions in ascending order. 
    The last element in the list does not exceed `until`.
    
    Args:
        n (int) : number of ones
        until (int) : upper bound of list elements
    
    Returns:
        ([Int]) : the generated list
    """
    a = 2 ** n - 1
    perms = [num2binstr(a)]
    while(True):
        a = a + lowbit(a) + lowbit(1 + a // lowbit(a)) // 2 - 1
        if (a > until):
            break
        else:
            perms.append(num2binstr(a))
            
    max_length = len(perms[-1])
    perms = list(map(lambda s: ('0' * (max_length - len(s)) + s), perms))
    return perms

def all_config(num_so, num_e): 
    until = 2 ** (num_so) - 1
    n = num_e
    configs = []
    for i in generate_perms(n, until):
        configs.append(i)
    return configs


def default_labeling(configs):
    q2o = {}  ## qubit label to occupation numbers
    o2q = {}  ## occupation numbers to qubit label
    num_qubits = ceil(log2(len(configs)))
    for idx, config in enumerate(configs):
        q_label = '0' * (num_qubits - len(num2binstr(idx))) + num2binstr(idx)
        q2o[q_label] = config
        o2q[config] = q_label
    return q2o, o2q

def reverse_labeling(configs):
    q2o = {}  ## qubit label to occupation numbers
    o2q = {}  ## occupation numbers to qubit label
    num_qubits = ceil(log2(len(configs)))
    for idx, config in enumerate(configs):
        idx = 2**num_qubits - idx -1
        q_label = '0' * (num_qubits - len(num2binstr(idx))) + num2binstr(idx)
        q2o[q_label] = config
        o2q[config] = q_label
    return q2o, o2q

def possible_excitations(num_so):
    excitations = set()
    for i in range(num_so):
        for j in range(i+1):
            excitations.add((i, j))
    return excitations


def label2Pauli(s):
    """
    Convert a Pauli string into Pauli object. 
    Note that the qubits are labelled in descending order: 'IXYZ' represents I_3 X_2 Y_1 Z_0
    
    Args: 
        s (str) : string representation of a Pauli term
    
    Returns:
        qiskit.quantum_info.Pauli: Pauli object of s
    """
    
    xs = []
    zs = []
    label2XZ = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
    for c in s[::-1]:
        x, z = label2XZ[c]
        xs.append(x)
        zs.append(z)
    return Pauli(z = zs, x = xs)


def get_entry_operators(num_so, num_e, labeling_method, mode):
    """
        Entry-operators are tensor products of some of the following operators: ['+', '-', '1', '0']. 
        '+' and '-' are qubit creation/annihilation operators, respectively. 
        '1' and '0' are counting operators for qubit state 1 and 0, respectively.
        Note that '+' = (X-iY)/2, '-' = (X+iY)/2, '1' = (I-Z)/2, '0' = (I+Z)/2.

        This function considers all possible excitations and calculates corresponding excitation oprators
        in terms of entry-operators. 

        Args:
            num_so (int) : Number of spin-orbitals(must be even).
            num_e (int) : Number of electrons(must be even).
            labeling_method (function) : 
                It maps each electron occupation configuration to a qubit computational basis state.
                By default it uses `default_labeling` method. 
                If it is set to `None`, default method will be used. 
                Please make sure that the input/output types match that of the `default_labeling` function.
        Returns:
            (tuple(int, dict(tuple(int, int), list(tuple(float, str))))):
                First nubmer in the tuple is total required number of qubits. 
                The second element is the mapping dictionary whose keys are indices of excitations
                ((i, j)->'a_i^ a_j'), and its values are list of entry-operators(str) with corresponding
                coefficients(float).

    """
    hopping_in_entry = {('0', '0'): '0',
                        ('1', '1'): '1',
                        ('0', '1'): '+',
                        ('1', '0'): '-'} ## (from, to)
    if mode == 'rhf':
        mapping = {}
        configs = all_config(num_so // 2, num_e // 2)
        half_qubits = int(ceil(log2(len(configs))))
        q2o, o2q = labeling_method(configs)
        excitations = possible_excitations(num_so // 2)

        def dual(s):
            return s[half_qubits:] + s[:half_qubits]

        for i, j in excitations:
            mapping[(i, j)] = []
            mapping[(i + num_so//2, j + num_so//2)] = []
            for c in configs:
                N = len(c)
                creation = N - i - 1
                annihilation = N - j - 1
                if c[creation] == '0' and c[annihilation] == '1':
                    hopped_c = c[: creation] + '1' + c[creation + 1 : annihilation] + '0' + c[annihilation + 1 :]
                    parity = 1 - 2 * (sum([int(i) for i in c[creation + 1 : annihilation]]) % 2)
                    entry_op = ''
                    for q0, q in zip(o2q[c], o2q[hopped_c]):
                        entry_op += hopping_in_entry[(q0, q)]
                    entry_op += ('I' * half_qubits) 
                    mapping[(i, j)].append((parity, entry_op))
                    mapping[(i + num_so//2, j + num_so//2)].append((parity, dual(entry_op)))
                elif creation == annihilation and c[creation] == '1':
                    entry_op = ''
                    for q in o2q[c]: 
                        entry_op += hopping_in_entry[(q, q)]
                    entry_op += ('I' * half_qubits)
                    mapping[(i, j)].append((1, entry_op))
                    mapping[(i + num_so//2, j + num_so//2)].append((1, dual(entry_op)))
        return 2 * half_qubits, mapping

    if mode == 'uhf' or mode == 'stacked_rhf':
        if mode == 'uhf':
            configs = all_config(num_so, num_e)
            excitations = possible_excitations(num_so)
        else :
            sub_configs = all_config(num_so // 2, num_e // 2)
            configs = []
            for i in sub_configs:
                for j in sub_configs:
                    configs.append(i + j)
            excitations = possible_excitations(num_so // 2)
            for p, q in excitations.copy():
                excitations.add((p + num_so // 2, q + num_so // 2))

        mapping = {}
        num_qubits = int(ceil(log2(len(configs))))
        q2o, o2q = labeling_method(configs)
        
        for i, j in excitations:
            mapping[(i, j)] = []
            for c in configs:
                N = len(c)
                creation = N - i - 1
                annihilation = N - j - 1
                if c[creation] == '0' and c[annihilation] == '1':
                    hopped_c = c[: creation] + '1' + c[creation + 1 : annihilation] + '0' + c[annihilation + 1 :]
                    parity = 1 - 2 * (sum([int(i) for i in c[creation + 1 : annihilation]]) % 2)
                    entry_op = ''
                    for q0, q in zip(o2q[c], o2q[hopped_c]):
                        entry_op += hopping_in_entry[(q0, q)]
                    mapping[(i, j)].append((parity, entry_op))
                elif creation == annihilation and c[creation] == '1':
                    entry_op = ''
                    for q in o2q[c]: 
                        entry_op += hopping_in_entry[(q, q)]
                    mapping[(i, j)].append((1, entry_op))
        return num_qubits, mapping


def operator2labels(operators_with_coef, index=0, num_qubits=0):
    # terminate the recursion when the pointer position (index) is larger than the length of the operator
    if index == num_qubits:
        return operators_with_coef
    # the matrix here means X, Y, Z, I, 0, 1, +, - at the specified pointer position (index)
    matrix = operators_with_coef[0][1][index]
    # go to the next loop when there's no 0, 1, +, - at the specified position
    if matrix in ['X', 'Y', 'Z', 'I']:
        return operator2labels(operators_with_coef, index+1, num_qubits)
    # define a new [coef, operator] list so that each of the [coef, operator] pair can be
    # separated into two terms with pauli matrix representation at the specified position
    new_operators_with_coef = []
    if matrix == '0':
        for [coef, operator] in operators_with_coef:
            first_op = operator[:index] + 'I' + operator[index+1:]
            second_op = operator[:index] + 'Z' + operator[index+1:]
            new_operators_with_coef.append([0.5*coef, first_op])
            new_operators_with_coef.append([0.5*coef, second_op])
    elif matrix == '1':
        for [coef, operator] in operators_with_coef:
            first_op = operator[:index] + 'I' + operator[index+1:]
            second_op = operator[:index] + 'Z' + operator[index+1:]
            new_operators_with_coef.append([0.5*coef, first_op])
            new_operators_with_coef.append([-0.5*coef, second_op])
    elif matrix == '+':
        for [coef, operator] in operators_with_coef:
            first_op = operator[:index] + 'X' + operator[index+1:]
            second_op = operator[:index] + 'Y' + operator[index+1:]
            new_operators_with_coef.append([0.5*coef, first_op])
            new_operators_with_coef.append([-0.5j*coef, second_op])
    elif matrix == '-':
        for [coef, operator] in operators_with_coef:
            first_op = operator[:index] + 'X' + operator[index+1:]
            second_op = operator[:index] + 'Y' + operator[index+1:]
            new_operators_with_coef.append([0.5*coef, first_op])
            new_operators_with_coef.append([0.5j*coef, second_op]) 
    return operator2labels(new_operators_with_coef, index+1, num_qubits)


# add the `coef` parameter
def operator2WeightedPauliOperator(operator, coef=1, num_qubits=None):
    pauli_labels = operator2labels([[coef, operator]], num_qubits=num_qubits)
    return WeightedPauliOperator([[coef, label2Pauli(label)] for (coef, label) in pauli_labels])

def get_naive_mapping_from_entry(entry_operators, num_qubits=None):
    """
    Get the naive mapping from a given entry_operator dictionary 

    Returns:
        naive_mapping (dict((tuple(int, int)), qiskit.quantum_info.Pauli)): 
            mapping, which should be passed into `complete_mapping`, 
            from some of the allowed a_p^ a_q to Pauli operator
    """
    naive_mapping = {}
    for (i, j) in entry_operators.keys():
        for (coef, operator) in entry_operators[(i, j)]:
            if (i, j) not in naive_mapping:
                naive_mapping[(i, j)] = operator2WeightedPauliOperator(operator, coef, num_qubits)
            else:
                naive_mapping[(i, j)] += operator2WeightedPauliOperator(operator, coef, num_qubits)
    return naive_mapping

def complete_mapping(naive_mapping):
    mapping = copy(naive_mapping)
    for k, w in naive_mapping.items():
        k1, k2 = k
        if (k1 != k2) and ((k2, k1) not in naive_mapping.keys()):
            new_k = (k2, k1)
            paulis = list(map(lambda x: [np.conj(x[0]), x[1]], w.paulis))
            mapping[new_k] = WeightedPauliOperator(paulis)
    return mapping

def fermionic2QubitMapping(num_so, num_e, labeling_method = default_labeling, mode = 'rhf'):
    """
        Generate qubit mapping for given number of spin-orbitals and electrons 
        based on certain labeling method(from occupation configurations to qubit states).

        Args:
            num_so (int) : Number of spin-orbitals, must be even as we only consider RHF configuration.
            num_e (int) : Number of electrons, must be even.
            labeling_method (function) : 
                It maps each electron occupation configuration to a qubit computational basis state.
                By default it uses `default_labeling` method. 
                If it is set to `None`, default method will be used. 
                Please make sure that the input/output types match that of the `default_labeling` function.

        Returns:
            (dict(tuple(int, int), WeightedPauliOperator)) : 
                A dictionary whose keys are indices of excitation operators((i, j)->'a_i^ a_j') 
                and values are corresponding Pauli operators(WeightedPauliOperator).
                
    """
    if labeling_method == None: labeling_method = default_labeling
    num_qubits, entry_operators = get_entry_operators(num_so, num_e, labeling_method, mode)
    # print(entry_operators)
    naive_mapping = get_naive_mapping_from_entry(entry_operators, num_qubits)
    mapping = complete_mapping(naive_mapping)
    return mapping
