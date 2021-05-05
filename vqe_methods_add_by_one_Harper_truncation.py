import scipy
import math

import pickle
import numpy as np

# import cirq
# import openfermioncirq
# from openfermioncirq import trotter
import scipy.sparse.linalg

import operator_pools
from tVQE import tUCCSD

#from openfermion import *
#from pyscf_backend import *
#from of_translator import *

QubitNumber=10





def adapt_vqe(geometry,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-7,
        theta_thresh    = 1e-10,
        adapt_maxiter   = 400,
        Pool            =['YIIIIIII'],
        Resultat        =[], 
        bond_legth      =0

        ):

    #Basis set has to be something recognized by PySCF.  (sto-3g, 6-31g, etc.)
    basis = "sto-3g"
    
    #I only trust RHF right now- I can debug open-shell refs if you need them though.
    reference = "rhf"
    
    #Number of frozen electrons.  Probably an even number.
    frozen_core = 2
    
    #Number of frozen virtual orbitals.
    frozen_vir = 0
    try:
        file = open('MolecularHamiltonians/MolecularHamiltonian_{}'.format(bond_legth),'rb')
        objectsFromFile = pickle.load(file)
        
        _0body_H = objectsFromFile[0][0]
        _1body_H = objectsFromFile[0][1]
        _2body_H = objectsFromFile[0][2] 
        _1rdm = objectsFromFile[0][3] 
        hf_energy = objectsFromFile[0][4] 
 
        
        H = objectsFromFile[1][0]
        ref = objectsFromFile[1][1]
        N_qubits = objectsFromFile[1][2] 
        S2 = objectsFromFile[1][3] 
        Sz = objectsFromFile[1][4] 
        Nop = objectsFromFile[1][5] 
        
        N_e = int(np.trace(_1rdm))
    except:
        
        _0body_H, _1body_H, _2body_H, _1rdm, hf_energy = get_integrals(geometry, basis, reference, frozen_core = frozen_core, frozen_vir = frozen_vir)
        N_e = int(np.trace(_1rdm))
        #Use OpenFermion to build sparse matrix representations of JW-transformed operators:
        H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(_0body_H, _1body_H, _2body_H, N_e)
        with open('MolecularHamiltonians/MolecularHamiltonian_{}'.format(bond_legth) , 'wb') as myHandle:
            pickle.dump(((_0body_H, _1body_H, _2body_H, _1rdm, hf_energy),(H, ref, N_qubits, S2, Sz, Nop)), myHandle, protocol=pickle.HIGHEST_PROTOCOL) 
    #, ref, N_qubits, S2, Sz, Nop)
    #H is the Hamiltonian, ref is the HF reference in the active space.  S2, S_z, and N_op are the S^2, S_z, and number operators if you happen to want them.
    
    ci = np.linalg.eigh(H.todense())[0][0]
    print(f"Problem reduced to {N_qubits} qubits.")
    print(f"HF energy:                {hf_energy:20.16f}")
    print(f"Active space FCI energy:   {ci:20.16f}")
    print('n_electrons:',N_e)
    print(ref.transpose().conj().dot(S2.dot(ref))[0,0].real)



    print('molecule.n_orbitals:', N_qubits//2)
    
    
    print(" adapt threshold:", adapt_thresh)
    print(" theta threshold:", theta_thresh)

    pool=operator_pools.OperatorPool(Pool)  
    reference_ket, hamiltonian=ref, H
    



    HamiltonianSupport=set(['0000000011','0000001100','0000110000','0001000010','0010000001','0011000000','0100000010','0110000000','1000000001','1001000000','1100000000'])











    w, v = scipy.sparse.linalg.eigs(hamiltonian,k=1, which='SR')   
    GSE = min(w).real
    a=v.transpose()[0]
    a.real[abs(a.real) < 1e-10] = 0.0
    a.imag[abs(a.imag) < 1e-10] = 0.0
    #print(scipy.sparse.csr_matrix(a))
    cx = scipy.sparse.coo_matrix(v.transpose()[0])
    CompareSet=set()
    for i in cx.col:
        tmp=bin(i)[2:]
        CompareSet.add((10-len(tmp))*'0'+tmp)
    if CompareSet.difference(HamiltonianSupport) or HamiltonianSupport.difference(CompareSet):
        print(CompareSet.difference(HamiltonianSupport) )
        print( HamiltonianSupport.difference(CompareSet))
        print(len(HamiltonianSupport))
        print(len(CompareSet))
        raise NameError('Unlucky error, Openfermion flipped orbitals, pool must be swaped') 
    
        
    print('Ground state energy:', GSE)

    #Thetas
    parameters = []
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    

    E = reference_ket.transpose().conj().dot(hamiltonian.dot(reference_ket))[0,0].real
    print('initial energy', E)

    print(" Start ADAPT-VQE algorithm")

    curr_state = 1.0*reference_ket

    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops


    error=[]
    trial_model=None
    curr_energy=float(100)
    print(" Now start to grow the ansatz")
    flag=0
    for n_iter in range(0,adapt_maxiter):
        
    
        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)                 
        print(" --------------------------------------------------------------------------")

        min_options = {'gtol': theta_thresh, 'disp':False}
        
        if flag==0:
            max_derivative=-math.inf
            trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)
            for op in range(n_ops):
                pos=0
                der=trial_model.derivative(parameters,pos,spmat_ops[op])
                #print(der)
                if der > max_derivative:
                    max_derivative=der
                    ansatz_ops_res=ansatz_ops.copy()
                    ansatz_mat_res=ansatz_mat.copy()
                    parameters_res=parameters.copy()
                    ansatz_ops_res.insert(pos,Pool[op])
                    ansatz_mat_res.insert(pos,spmat_ops[op])
                    parameters_res.insert(pos,0)
                        
                        
       
            
            ansatz_mat=ansatz_mat_res
            ansatz_ops=ansatz_ops_res
            parameters=parameters_res
            trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
            options = min_options, method = 'BFGS', callback=trial_model.callback)
            
            parameters=list(opt_result['x'])
            print(ansatz_ops)
            
            curr_state = trial_model.prepare_state(parameters)
            curr_energy = trial_model.energy(parameters)
            if  abs(curr_energy-GSE)>adapt_thresh:
                #print(" new state: ",curr_state)
                print(" Finished: %20.12f" % curr_energy)
                print(" Error: %20.12f" % abs(curr_energy-GSE))
                print(bond_legth)
                error.append(abs(curr_energy-GSE))
                continue
            else:
                print(" Finished: %20.12f" % curr_energy)
                print(" Error: %20.12f" % abs(curr_energy-GSE))
                error.append(abs(curr_energy-GSE))
                flag=1
                break

    Resultat.append({'bond_length:':bond_legth, 'Pool:':Pool,'ansatz:':ansatz_ops,'parameters:':parameters,
                     'final_error:':abs(curr_energy-GSE),'GSE:':GSE,
                     'HF:': hf_energy,'error_before_#_step:':error})
    print('\n','\n','------------------------------------------','\n')
                        
                        
            
        

