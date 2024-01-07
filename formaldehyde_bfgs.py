import pennylane as qml
import pennylane.numpy as np
from single_point import matrixexponen
from pennylane.qchem import active_space
from single_point import _complete_pauli_string
from scipy import optimize
from scipy.signal import find_peaks
from pennylane import RMSPropOptimizer, AdagradOptimizer, AdamOptimizer
from BFGS import BFGSOptimizer

symbols = ["C", "O", "H", "H"]


geometry =  1.88973*np.array([[-0.000000,   0.000019,  -0.530799],  
   [-0.000000,   0.000019,   0.685924],   
   [-0.000000,  -0.926370,  -1.126440],   
   [-0.000000,   0.926331,  -1.126410]], requires_grad = True) ### HF optimized 


#geometry = np.array(np.load("ammonia_bfgs_es_11qubits_+_geometry.npy", allow_pickle = True)[-1], requires_grad = True) ### restart from last iteration

dev = qml.device("default.qubit", wires=21)

mol = qml.qchem.Molecule(symbols, geometry, basis_name = 'sto-3g')

core, active = active_space(16, orbitals=12, active_electrons=2, active_orbitals=4)  ### electron, spatial orbitals, active_electrons, active spatial orbitals

num_qubits = 13

configuration_1 = [1, 1, 0, 0, 0, 0, 0, 0]
#configuration_2 = [0, 1, 1, 0, 0, 0, 0, 0]

str_1 = ""
#str_2 = ""

for k in range(len(configuration_1)):
    str_1 += str(configuration_1[k])
#    str_2 += str(configuration_2[k])

index_1 = int(str_1, base = 2)

#index_2 = int(str_2, base = 2)

initial_array = np.zeros(2**8)

initial_array[index_1] = 1
#initial_array[index_2] = -1/np.sqrt(2)

bitstrings = np.linspace(0, 1, 2**num_qubits)

def check_if_nan(matrix):
    shape = np.shape(matrix)
    flattened_array = matrix.flatten()
    for k in range(len(flattened_array)):
        if np.isnan(flattened_array[k]):
            flattened_array[k] = 0.0
    new_matrix = np.reshape(flattened_array, shape)
    return new_matrix

def check_phase_shift(energy, prev_energy, gradient, stepsize, width_of_the_box):
    g_norm = np.linalg.norm(gradient)
    approx_dphi = 5 * (energy - prev_energy) * g_norm * stepsize
    print(approx_dphi)
    if abs(approx_dphi) > width_of_the_box / 2**num_qubits:
        return False
    else:
        return True

@qml.qnode(dev)
def circuit1(weights):
   # qml.BasisState(np.array([1, 0, 0, 0, 0, 1, 0, 0]), wires = [0, 1, 2, 3, 4, 5, 6, 7])
    qml.templates.MottonenStatePreparation(initial_array, wires = [0, 1, 2, 3, 4, 5, 6, 7])
    hamiltonian = qml.qchem.diff_hamiltonian(mol, core = core, active = active)(weights[:])
    matrix = 0
    for coeff, op in zip(hamiltonian.coeffs, hamiltonian.ops):
        op = _complete_pauli_string(op, 8)
        matrix += coeff * op.matrix
    matrix = matrix
    unitary = matrixexponen(np.shape(matrix)[0], -1j*matrix)
    qml.templates.qpe.QuantumPhaseEstimation(unitary, target_wires = [0, 1, 2, 3, 4, 5, 6, 7], estimation_wires = range(8, num_qubits+8))
    return qml.probs(wires=range(8, num_qubits+8))

def softmax(x, temperature = 0.035):
    softmaxed = np.exp(x / temperature) / np.sum(np.exp(x / temperature), axis=0)
    return softmaxed

def find_highest_peak(probabilities, temperature, sm_param):
    softmaxed = softmax(probabilities, temperature)
    categorical_vector = np.multiply(np.arange(2**num_qubits), softmaxed)
    category = np.sum(categorical_vector)
    g_filter = boxcar_filter(2**num_qubits, temperature, category)
    return g_filter


def boxcar_filter(length_of_filter, width_of_the_box, center_of_the_box):
    k = 1000
    filter_array = 0.5 + 0.5 * np.tanh(k * (np.linspace(0, 1, length_of_filter) - (center_of_the_box - width_of_the_box)/length_of_filter)) - (0.5 + 0.5 * np.tanh(k * (np.linspace(0, 1, length_of_filter) - (center_of_the_box + width_of_the_box)/length_of_filter)))
    return filter_array
    
def function4(weights, temperature, sm_param, phase_check = False, remove_peak = False):
    probabilities = circuit1(weights)
    g_filter = find_highest_peak(probabilities, temperature, sm_param)
    if remove_peak:
        print("I am removing the peak")
        peak_to_remove = np.multiply(probabilities, g_filter)
        new_probabilities = probabilities - peak_to_remove
        g_filter = find_highest_peak(new_probabilities, temperature, sm_param)
    final_signal = np.multiply(probabilities, g_filter)
    trig_moment = np.sum(np.multiply(final_signal, np.exp(1j * ((2 * np.pi/ 2**num_qubits)) * np.arange(2**num_qubits))))
    energy_as_mean_phase_direction =  - np.angle(trig_moment) + mol.offset[0]
    if phase_check:
        return probabilities, energy_as_mean_phase_direction
    else:
        return energy_as_mean_phase_direction

def grad_fun(geometry, temperature, width, phase_check = False, remove_peak = False):
    grad_geom = qml.grad(function4)(geometry, temperature, width, phase_check, remove_peak)
    grad = check_if_nan(grad_geom)
    return grad
    
def step_and_follow_root(function4, geometry,  temperature, width, grad_fn=grad_fun, prev_energy = None, remove_peak = False):
   # print("Miao")
    new_geometry, energy = opt.step_and_cost(function4, geometry, temperature, width, grad_fn=grad_fun, remove_peak=remove_peak)
   # print(new_geometry)
   # print(energy)
    if prev_energy == None:
        print("prev_energy is  :  " + str(prev_energy))
        return new_geometry, energy
    else:
        keep_following = check_phase_shift(energy, prev_energy, opt.old_grad, opt.stepsize, width)
    if keep_following:
        print("Following the right root!")
        return new_geometry, energy
    else:
        print("Recalculating gradient, wrong root was chosen...")
        mol = qml.qchem.Molecule(symbols, geometry, basis_name = 'sto-3g')
        #### here working with default optimizers step_and_follow_root(blbalbla)
        new_geometry, energy = opt.step_and_cost(function4, geometry,  temperature, width, grad_fn=grad_fun, remove_peak = True)
        return new_geometry, energy
    
    
    
energy = []

geometries = []

F = []

probabilities = []

width = np.asarray([4.0], requires_grad = False)

energy_ini = function4(geometry, 0.01, 4)
energy.append(energy_ini)

temperature = np.asarray([0.01], requires_grad = False)

#opt = AdamOptimizer()
opt = BFGSOptimizer()
prev_energy = None
for j in range(15):
    print("Inside the opt loop", flush = True)
    geometry, energy_at_step = opt.step_and_cost(function4, geometry, temperature, width, grad_fn=grad_fun)
#    geometry, energy_at_step = step_and_follow_root(function4, 
#                                      geometry,
#                                      temperature,
#                                      width,
#                                      grad_fn=grad_fun,
#                                      prev_energy = prev_energy)
 #   geometry, temperature, width = params
    # geometry, temperature, width, phase_check, remove_peak = params
    mol = qml.qchem.Molecule(symbols, geometry, basis_name = 'sto-3g') 
    geometries.append(geometry)
    prev_energy = energy_at_step
    energy.append(energy_at_step)
    F.append(opt.old_grad)
    print(j, flush = True)
    print("Geometry is:  " + str(geometry[-1][:]), flush = True)
    print("Energy is:   " + str(energy[-1]), flush = True)
    print("Forces are:   " + str(F[-1]), flush = True)
    if j % 10 == 0:
        probability_distribution, energy_at_step = function4(geometry, temperature, width, True, False)
        print("Peaks are located at   :   " + str(find_peaks(probability_distribution)))
        probabilities.append(probability_distribution)
        print(f'n: {j}, E: {energy[-1]:.8f}, Force-max: {abs(opt.old_grad).max():.8f}')

np.save("formald_bfgs_gs_13_qubits_energy", energy)

np.save("formald_bfgs_gs_13_qubits_geometry", geometries)

np.save("formald_bfgs_gs_13_qubits_forces", F)

np.save("formald_bfgs_gs_13_qubits_probabilities", probabilities)


