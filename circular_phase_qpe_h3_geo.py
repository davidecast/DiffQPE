import pennylane as qml
import pennylane.numpy as np
from single_point import matrixexponen
from single_point import _complete_pauli_string



symbols = ["H", "H", "H"]

geometry = 1.88973*np.array([[ 0.0, 0.0, 0.0], [1.0, 1.0, 0.0],
     [2.0, 0.0, 0.0]], requires_grad=True)

dev = qml.device("default.qubit", wires=16)

num_qubits = 13

mol = qml.qchem.Molecule(symbols, geometry)

bitstrings = np.linspace(0, 1, 2**num_qubits)

def mod(a, m):
    return a - m * np.floor(a/m)

@qml.qnode(dev)
def circuit1(weights):
    n_electrons = 2
    H = qml.qchem.diff_hamiltonian(mol)(weights[:])
    generators = qml.symmetry_generators(H)
    paulixops = qml.paulix_ops(generators, len(H.wires))
    paulix_sector = qml.qchem.optimal_sector(H, generators, n_electrons)
    H_tapered = qml.taper(H, generators, paulixops, paulix_sector)
    for k in range(3):
        qml.PauliX(wires = k)
    matrix = 0
    for coeff, op in zip(H_tapered.coeffs, H_tapered.ops):
        op = _complete_pauli_string(op, 3)
        matrix += coeff * op.matrix
    unitary = matrixexponen(np.shape(matrix)[0], -1j*matrix)
    qml.templates.qpe.QuantumPhaseEstimation(unitary, target_wires = [0, 1, 2], estimation_wires = range(3, num_qubits+3))
    return qml.probs(wires=range(3, num_qubits+3))

def softmax(x, temperature = 0.035):
    softmaxed = np.exp(x / temperature) / np.sum(np.exp(x / temperature), axis=0)
    return softmaxed


def boxcar_filter(length_of_filter, width_of_the_box, center_of_the_box):
    k = 10000
    filter_array = 0.5 + 0.5 * np.tanh(k * (np.linspace(0, 1, length_of_filter) - (center_of_the_box - width_of_the_box)/length_of_filter)) - (0.5 + 0.5 * np.tanh(k * (np.linspace(0, 1, length_of_filter) - (center_of_the_box + width_of_the_box)/length_of_filter)))
    return filter_array
    
def function4(weights, temperature, sm_param):
    probabilities = circuit1(weights)
    softmaxed = softmax(probabilities, temperature)
    categorical_vector = np.multiply(np.arange(2**num_qubits), softmaxed)
    category = np.sum(categorical_vector)
    g_filter = boxcar_filter(2**num_qubits, sm_param, category)
    final_signal = np.multiply(probabilities, g_filter)
    trig_moment = np.sum(np.multiply(final_signal, np.exp(1j * ((2 * np.pi/ 2**num_qubits)) * np.arange(2**num_qubits))))
#    shifted_mpd = mod(np.angle(trig_moment)/2 * np.pi, 2 * np.pi)
    energy_as_mean_phase_direction =  - np.angle(trig_moment) 
    return energy_as_mean_phase_direction


energy = []

geometries = []

F = []

width = 10

print(function4(geometry, 0.0035, width), flush = True)

#for k in range(150):
#    forces = -qml.grad(function4, argnum = 0)(geometry, 0.0035, width)
#    geometry = geometry + 0.5 * forces
#    mol = qml.qchem.Molecule(symbols, geometry) 
#    geometries.append(geometry)
#    energy.append(function4(geometry, 0.0035, width))
#    F.append(forces)
#    print(k, flush = True)
#    print("Geometry is:  " + str(geometry[-1]), flush = True)
#    print("Energy is:   " + str(energy[-1]), flush = True)
#    print("Forces are:   " + str(F[-1]), flush = True)
#    if k % 5 == 0:
#        print(f'n: {k}, E: {energy[-1]:.8f}, Force-max: {abs(forces).max():.8f}')


#np.save("energies_h2_geo_opt_circ_phase", energy)
#
#np.save("geometries_h2_geo_opt_circ_phase", geometries)
#
#np.save("forces_h2_geo_opt_circ_phase", F)

