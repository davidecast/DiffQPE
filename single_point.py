import pennylane as qml
from autograd import grad
import pennylane.numpy as np
from pennylane import hf
from scipy.linalg import expm
#from torch.linalg import matrix_exp

#symbols = ["H", "H", "H"]

E = -1.2744376574936744
# equilibrium geometry
#geometry = 1.88973*np.array([[ 0.0056957528, 0.0235477326, 0.0000000000], [0.5224540730, 0.8628715457, 0.0000000000],
#     [0.9909500019, -0.0043172515, 0.0000000000]], requires_grad=True)

 # non-equilibrium geometry
#geometry = 1.88973*np.array([[ 0.0, 0.0, 0.0], [1.0, 1.0, 0.0],
# [2.0, 0.0, 0.0]], requires_grad=True)

#target_wires = [0, 1, 2, 3, 4, 5]
#estimation_wires = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# basis set exponents
#alpha = np.array([[3.4253, 0.6239, 0.1689], [3.4253, 0.6239, 0.1689],
# [3.4253, 0.6239, 0.1689]], requires_grad=True)

 # basis set contraction coefficients
#coeff = np.array([[0.1543, 0.5353, 0.4446], [0.1543, 0.5353, 0.4446],
# [0.1543, 0.5353, 0.4446]], requires_grad=True)


#dev = qml.device("lightning.qubit", wires=19)  ### not working with lightning?

def cost_function(mol, *args, target_wires = range(6), estimation_wires = range(6,19)):
#    value = np.pi * (1 - np.argmax(energy(mol)(*args))) * (2**len(np.arange(6, 16)))**-1
#    estimated_phase = 1 - np.argmax(energy(mol)(*args))
    estimated_phase = np.dot(np.arange(2**13), softmax(energy(mol)(*args)))
    estimated_phase /= 2**13
 #   value = (4 * np.pi * (1 - estimated_phase)) / ((2**len(np.arange(6, 16))))
    value = - 2 * np.pi * (estimated_phase)
    print(value)
#    eigval = np.exp(1j*value)
#    final_estimate = np.arccos(np.real(eigval))
    return value

def softmax(x):
    temperature = 0.035
    softmaxed = np.exp(x / temperature) / np.sum(np.exp(x / temperature), axis=0)
    return softmaxed

def _complete_pauli_string(operator, n_of_final_wires):
    n_of_paulis = len(operator.wires)
    if n_of_paulis == n_of_final_wires:
        new_operator = operator
    else:
        operator = qml.operation.Tensor(operator)
        initial_wires = operator.wires.tolist()
        final_wires = np.arange(n_of_final_wires).tolist()
        missing_wires = list(set(final_wires).difference(initial_wires))
        new_operator = qml.Identity(wires = [0])
        pauli_from_operator = 0
        append_identity = 0
        for wire in np.arange(n_of_final_wires):
            if wire in initial_wires:
                new_operator = new_operator @ operator.obs[pauli_from_operator]
                pauli_from_operator += 1
            else:
                new_operator = new_operator @ qml.Identity(wires = missing_wires[append_identity])
                append_identity += 1
    return new_operator

def check_exp_diff(mol, *args):
    hamiltonian = qml.qchem.diff_hamiltonian(mol)(*args[:])
    matrix = 0
    for coeff, op in zip(hamiltonian.coeffs, hamiltonian.ops):
        op = _complete_pauli_string(op, 6)
        matrix += coeff * op.matrix
   # second_unitary = other_matrix_exponential(matrix)
    second_unitary =  matrixexponen(np.shape(matrix)[0], -1j*matrix)
    value = np.real(np.sum(second_unitary))
    return value


def other_matrix_exponential(matrix):       #### NOT Differentiable
    eigval, eigvec = np.linalg.eigh(matrix)
    new_matrix = eigvec @ np.diag(np.exp(-1j*eigval)) @ np.transpose(np.conj(eigvec))
    return new_matrix

def matrixexponen(n,a):     ###### check_exp_diff returns reasonable gradient 
    q = 6
    a2 = a
    a_norm = max(sum(abs(a)))   ### check for differentiable max
    ee = (np.log2( a_norm)) + 1
    s = np.ceil(max( 0, ee + 1 ))  ###   check for differentiable ceiling
    a2 = a2 / ( 2.0 ** s )
    x = a2
    c = 0.5
    e = np.eye( n, dtype = np.complex64 ) + c * a2
    d = np.eye( n, dtype = np.complex64 ) - c * a2
    p = True
    for k in range ( 2, q + 1 ):
        c = c * float( q - k + 1 ) / float( k * ( 2 * q - k + 1 ) )
        x = np.dot( a2, x )
        e = e + c * x
        if ( p ):
            d = d + c * x
        else:
            d = d - c * x
        p = not p
            #  E -> inverse(D) * E
    e = np.linalg.solve(d, e)
            #  E -> E^(2*S)
    for k in range ( 0, int(s.data.tolist())):   ####
        e = np.dot ( e, e )
    return e


def energy(mol):
    @qml.qnode(dev, diff_method = "parameter-shift")
    def circuit(*args, target_wires, estimation_wires):
     #   target_wires = np.array([0, 1, 2, 3, 4, 5])
     #   estimation_wires = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        qml.BasisState(np.array([1, 1, 0, 0, 0, 0]), wires = target_wires) #### if we want to compute something different from the ground state energy we should modify this initialization (perhaps suggesting a QST every tot step of a geometry optimization procedure)
        hamiltonian = qml.qchem.diff_hamiltonian(mol)(*args[:])
        #eigval, eigvec = np.linalg.eigh(qml.utils.sparse_hamiltonian(hamiltonian).toarray())
        #print(eigval[:5])
        #print(eigvec)
        #print("Minimum eigval is   :" + str(np.min(np.linalg.eigh(qml.utils.sparse_hamiltonian(hamiltonian).toarray()))))
        matrix = 0
        for coeff, op in zip(hamiltonian.coeffs, hamiltonian.ops):
            op = _complete_pauli_string(op, 6)
            matrix += coeff * op.matrix
        #print("Minimum eigval after transformation is   :"  + str(np.min(np.linalg.eigh(matrix))))
        unitary = matrixexponen(np.shape(matrix)[0], -1j*matrix)
  #      second_unitary = other_matrix_exponential(matrix)
#        unitary = expm(-1j*matrix)
        qml.templates.qpe.QuantumPhaseEstimation(unitary, target_wires = target_wires, estimation_wires = estimation_wires)
        return qml.probs(wires = estimation_wires)
    return circuit

#def finite_diff_grad(*args):
#    grad_fin_diff = qml.finite_diff(cost_function(mol))(*args)
#    return grad_finite_diff

#mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)

#print(mol.alpha)
#mol = hf.Molecule(symbols, geometry)
#args = [geometry, alpha, coeff]
#args = [geometry]
#print(qml.hf.hf_energy(mol)(*args))
#print(grad(qml.qchem.hf_energy(mol))(*args))
#evaluated_phase = cost_function(mol, *args, target_wires = target_wires, estimation_wires = estimation_wires)
#print(evaluated_phase)
#value = check_exp_diff(mol, *args)
#print(value)
#check_gradient = grad(check_exp_diff, argnum = 1)(mol, *args) works for matrixexponen
#circuit_function = energy(mol)(*args, target_wires = target_wires, estimation_wires = estimation_wires)
#print(circuit_function)
#print(np.sum(circuit_function))
#print(np.shape(circuit_function))
#print(type(circuit_function))
#check_circuit_diff = qml.jacobian(energy(mol), argnum = 0)(*args, target_wires = target_wires, estimation_wires = estimation_wires)
#print(check_circuit_diff)
#print(type(check_circuit_diff))
# gradient for nuclear coordinates
#finite_diff_force = - finite_diff_grad(*args)
#forces = -grad(cost_function, argnum = 1)(mol, *args, target_wires = target_wires, estimation_wires = estimation_wires)
# gradient for basis set exponents
#gradient_alpha = grad(cost_function, argnum = 2)(mol, *args)
# gradient for basis set contraction coefficients
#gradient_coeff = grad(cost_function, argnum = 3)(mol, *args)


#### jacobian instead of gradient for the matrix

#print("This is the estimated energy:   " + str(evaluated_phase))
#print("This is the ref. energy:    " + str(E))
#print("This is the gradient (finite_diff) w.r.t. the nuclear coordinates:    " + str(finite_diff_force))
#print("This is the gradient w.r.t. the nuclear coordinates:   " + str(forces))
#print("This is the gradient w.r.t. the basis set exponents:   " + str(gradient_alpha))
#print("This is the gradient w.r.t. the contractions:   " + str(gradient_coeff))
