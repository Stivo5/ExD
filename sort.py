
from triqs.operators.util.hamiltonians import *
from triqs.operators.util import *
from triqs.operators import *
from triqs.gf import *
from triqs.atom_diag import *

import sys
sys.path.append("/Users/ShuyuZhang/Documents/Projects/Cyrus/Codes/d_orbital_atom_Cyrus/defect_atom_diag/")
from def_atom_diag import *


import numpy as np
import sympy as sp


#*************************************************************************************
# Electron-hole correlation function in imaginary frequency
def get_suscep_lind_iw(beta,ad,spin_names,orb_names,eigensys,n_states,n_iw):

    '''
    Computes retarded electron-hole correlation function:

    \chi_{abcd}(\omega) = \frac{1}{Z} \sum_{m,n} \frac{e^{-\beta E_n} - e^{-\beta E_m}}{i\omega + E_n - E_m} \, \langle n | c_a^\dagger c_b | m \rangle \langle m | c_c^\dagger c_d | n \rangle   

    
    Input:
    beta: inverse temperature
    ad: atom diag object
    spin_names: List of spins
    orb_names: Orbital names  
    eigensys: Formatted solution to atom diag
    n_lambda: "Unoocupied states"
    n_alpha: "Occupied states," need to replace with Fermi weights
    n_iw: Imaginary frequency mesh number of points
    
    
    Output:
    L2w: Retarded electron-hole correlation function as a triqs GF object
    
    '''
    
    n_orb=len(orb_names)
    n_spin=len(spin_names)

    # kB=8.61733262e-5 # eV/K
    # beta = beta/kB

    # Set up Greens function object:
    iw_mesh = MeshImFreq(beta = beta, statistic = 'Boson', n_iw=n_iw)
    L2w = Gf(mesh=iw_mesh, target_shape=[n_spin*n_orb,n_spin*n_orb,n_spin*n_orb,n_spin*n_orb])

    den_mats=[]
    for alpha in range(min(n_states,len(eigensys))):
        
        for lam in range(min(n_states,len(eigensys))):
            
            # if alpha==lam: continue
            
            # Skip pairs of states with the same energies
            if eigensys[alpha][1] - eigensys[lam][1] < 1e-5: continue  

            # Convert from states in energy order to those in Hilbert space
            n_eig_a=int(eigensys[alpha][2])
            n_eig_l=int(eigensys[lam][2])
        
            # Get desired states in eigenvector basis
            state_eig_a = np.zeros((int(ad.full_hilbert_space_dim)))
            state_eig_a[int(n_eig_a)]=1.0
            state_eig_l = np.zeros((int(ad.full_hilbert_space_dim)))
            state_eig_l[int(n_eig_l)]=1.0
        
            # Construct density matrix
            den_mat=np.zeros((n_spin*n_orb,n_spin*n_orb),dtype='complex128')
            for s1 in range(0,n_spin):
                for s2 in range(0,n_spin):
                    for ii in range(0,n_orb):
                        for jj in range(0,n_orb):
                        
                            den_op=c_dag(spin_names[s1],orb_names[ii]) * c(spin_names[s2],orb_names[jj])

                            xx=ii+s1*n_orb
                            yy=jj+s2*n_orb
                        
                            den_mat[xx,yy]=np.dot(state_eig_a,act(den_op,state_eig_l,ad))

            #den_mat=np.eye(n_spin*n_orb)
            den_mats.append([eigensys[lam][1],eigensys[alpha][1],den_mat,np.einsum('ij,kl->ijkl',den_mat,np.conjugate(den_mat.T))])
            
    L_w=[]
    for iw in iw_mesh:
        L=0
        for dm in den_mats:
            L+=( (np.exp(-beta*dm[0])-np.exp(-beta*dm[1]))*dm[3]) / (1j * iw.value +dm[0]-dm[1])
            #L+=1j * dm[2] / (omega+dm[0]+1j*eta)

            # TEST
            #for el in np.ndarray.flatten(1j * (1-np.exp(dm[0]/(kB*T)))*dm[2] / (omega+dm[0]+1j*eta)):
            #    if not el==el:
            #        print('NAN!')
            #        print(omega,dm[0],(1-np.exp(dm[0]/(kB*T))),omega+dm[0]+1j*eta)
            #        raise
            
            
            #print((1-np.exp(dm[0]/(kB*T))),dm[0])
            #L+=1j * (1-np.exp(beta*dm[0])) / (omega+dm[0]+1j*eta)
            #print(dm[0])
        L_w.append(L)


    L_w=np.stack(L_w)
        
    for orb_1 in range(n_orb*n_spin):
        for orb_2 in range(n_orb*n_spin):
            for orb_3 in range(n_orb*n_spin):
                for orb_4 in range(n_orb*n_spin):
                    Gw=Gf(mesh=iw_mesh, data=L_w[:,orb_1,orb_2,orb_3,orb_4])
                    L2w[orb_1,orb_2,orb_3,orb_4]<<Gw
        
    return L2w,L_w, den_mats
        
#*************************************************************************************

#*************************************************************************************
# Electron-hole correlation function in imaginary frequency
def get_suscep_lind_w(beta,ad,spin_names,orb_names,eigensys,n_states,w_mesh,eta = 0.01):

    '''
    Computes retarded electron-hole correlation function:

    \chi_{abcd}(\omega) = \frac{1}{Z} \sum_{m,n} \frac{e^{-\beta E_n} - e^{-\beta E_m}}{i\omega + E_n - E_m} \, \langle n | c_a^\dagger c_b | m \rangle \langle m | c_c^\dagger c_d | n \rangle   

    
    Input:
    beta: inverse temperature
    ad: atom diag object
    spin_names: List of spins
    orb_names: Orbital names  
    eigensys: Formatted solution to atom diag
    n_lambda: "Unoocupied states"
    n_alpha: "Occupied states," need to replace with Fermi weights
    n_iw: Imaginary frequency mesh number of points
    
    
    Output:
    L2w: Retarded electron-hole correlation function as a triqs GF object
    
    '''
    
    n_orb=len(orb_names)
    n_spin=len(spin_names)

    # kB=8.61733262e-5 # eV/K
    # beta = beta/kB

    # Set up Greens function object:
    L2w = Gf(mesh=w_mesh, target_shape=[n_spin*n_orb,n_spin*n_orb,n_spin*n_orb,n_spin*n_orb])

    den_mats=[]
    for alpha in range(min(n_states,len(eigensys))):
        
        for lam in range(min(n_states,len(eigensys))):
            
            # Why can I not skip states with the same energy?????
            if alpha==lam: continue
            
            # Skip pairs of states with the same energies
            # if eigensys[alpha][1] - eigensys[lam][1] < 1e-5: continue  

            # Convert from states in energy order to those in Hilbert space
            n_eig_a=int(eigensys[alpha][2])
            n_eig_l=int(eigensys[lam][2])
        
            # Get desired states in eigenvector basis
            state_eig_a = np.zeros((int(ad.full_hilbert_space_dim)))
            state_eig_a[int(n_eig_a)]=1.0
            state_eig_l = np.zeros((int(ad.full_hilbert_space_dim)))
            state_eig_l[int(n_eig_l)]=1.0
        
            # Construct density matrix
            den_mat=np.zeros((n_spin*n_orb,n_spin*n_orb),dtype='complex128')
            for s1 in range(0,n_spin):
                for s2 in range(0,n_spin):
                    for ii in range(0,n_orb):
                        for jj in range(0,n_orb):
                        
                            den_op=c_dag(spin_names[s1],orb_names[ii]) * c(spin_names[s2],orb_names[jj])

                            xx=ii+s1*n_orb
                            yy=jj+s2*n_orb
                        
                            den_mat[xx,yy]=np.dot(state_eig_a,act(den_op,state_eig_l,ad))

            #den_mat=np.eye(n_spin*n_orb)
            den_mats.append([eigensys[lam][1],eigensys[alpha][1],den_mat,np.einsum('ij,kl->ijkl',den_mat,np.conjugate(den_mat.T))])
            
    L_w=[]
    for w in w_mesh:
        L=0
        
        for dm in den_mats:
            L+=( (np.exp(-beta*dm[0])-np.exp(-beta*dm[1]))*dm[3]) / (w + 1j*eta +dm[0]-dm[1])
        
        L_w.append(L)


    L_w=np.stack(L_w)
        
    for orb_1 in range(n_orb*n_spin):
        for orb_2 in range(n_orb*n_spin):
            for orb_3 in range(n_orb*n_spin):
                for orb_4 in range(n_orb*n_spin):
                    Gw=Gf(mesh=w_mesh, data=L_w[:,orb_1,orb_2,orb_3,orb_4])
                    L2w[orb_1,orb_2,orb_3,orb_4]<<Gw
        
    return L2w,L_w, den_mats
        
#*************************************************************************************


#*************************************************************************************
# Density-density susceptibility
def get_den(L2w):

    w_mesh=L2w.mesh
    chi=Gf(mesh=w_mesh, target_shape=L2w.target_shape[0:2])
    n_orb_n_spin=L2w.target_shape[0]
    
    for orb_1 in range(n_orb_n_spin):
        for orb_2 in range(n_orb_n_spin):
            chi[orb_1,orb_2] << L2w[orb_1,orb_2,orb_2,orb_1]

    chi_tot=Gf(mesh=w_mesh, target_shape=())
    chi_tot << 0
    
    for orb1 in range(n_orb_n_spin):
        for orb2 in range(n_orb_n_spin):
            chi_tot+=chi[orb1,orb2]
            
    return chi,chi_tot

#*************************************************************************************




#*************************************************************************************
# Print out information about the states
def sort_states_full(spin_names,orb_names,ad,verbose=True):
    '''
    Sort eigenstates and write them in fock basis
    
    Inputs:
    orb_names: List of orbitals
    spin_names: List of spins
    ad: Solution to the atomic problem

    Outputs:
    eigensys: List in the form [states, energies, eigenstate #]
    
    '''

    n_orb=len(orb_names)
    n_spin=len(spin_names)

    
    n_eigenvec=0.0
    eigensys=[]

    # Loop through subspaces
    for sub in range(ad.n_subspaces):
        
        # convert states in computational basis
        subspace_fock_state=[]
        print('states in subspace ', sub, ': ')
        for fs in ad.fock_states[sub]:
            state =int(bin(fs)[2:])
                
            state_len=n_orb*n_spin
            # to fill in the leading 0's in computational basis
            state_comp = f'{state:0{state_len}d}'
            subspace_fock_state.append(state_comp)
            print('|', state_comp, '>')
       

        u_mat=ad.unitary_matrices[sub].conj().T
        st_mat=subspace_fock_state
            
        # store state and energy    
        for ind in range(ad.get_subspace_dim(sub)):

            eng=ad.energies[sub][ind]
            eigensys.append([[u_mat[ind,:],st_mat],eng,int(n_eigenvec)])

            # Keep track of the absolute number of eigenstate
            n_eigenvec += 1

    # Sort by energy from smallest to largest
    eigensys.sort(key=take_second)

    if verbose:
        print('Finished sorting states')

    return eigensys
#*************************************************************************************





#*************************************************************************************
# Tune chemical, generate ad and dm
def solve_ad_full(H, fops, mu_in, verbose=False):
    '''
    Solve the atomic problem and tune the chemical potential to the
    target filling. Very inefficient way of doing it but works for now.

    Inputs:
    H: Hamiltonian operator
    spin_names: List of spins
    orb_names: List of orbitals
    fops: Many-body operators
    mu_in: Dictionary for chem pot options
    verbose: write stuff out 
    
    Outputs:
    ad: Solution to atomic problem
    mu: Chemical potential
    '''
    
    mu_init=mu_in['mu_init']
    tune_occ=mu_in['tune_occ']
    target_occ=mu_in['target_occ']
    const_occ=mu_in['const_occ']
    step=mu_in['mu_step']
    
    if tune_occ and const_occ:
        print('ERROR: You have to choose between tuning and constraining the occupation.')
        sys.exit(1)
    elif not tune_occ and not const_occ:
        print('WARNING: Assuming tune_occ')

    

    # Setup the particle number operator
    N = Operator()
    for op in fops:
        N += n(op[0],op[1])

    filling=0
    mu=mu_init

    # Constrained occupation:
    if const_occ:
        ad = AtomDiagComplex(H, fops, n_min=int(target_occ), n_max=int(target_occ))

        
        
    # Tune the chemical potential so that the group state occupancy hits target_occ
    elif tune_occ:
        while True:

            # Add chemical potential
            H += mu * N
                        
            # Compute ground state occupancy by taking beta >> 1
            beta = 1e5
            ad = AtomDiagComplex(H, fops)
            dm = atomic_density_matrix(ad, beta)
            filling = trace_rho_op(dm, N, ad)

            print("mu:",mu,"filling:",filling)

            if abs(filling.real-target_occ) < 1.0e-4:
                break
            elif filling.real < target_occ:
                H += -mu * N
                mu+=-step
            elif filling.real > target_occ:
                H += -mu * N
                mu+=step
                
    else: # Use input chemical potential
        H += mu * N
        ad = AtomDiagComplex(H, fops)

    beta = 1e10
    dm = atomic_density_matrix(ad, beta)
    filling = trace_rho_op(dm, N, ad)

    if not const_occ:
        print("Chemical potential: ",mu)

    if verbose:
        print('# of e- in groud state: ', filling)


    return ad, H, mu
#*************************************************************************************





#*************************************************************************************
# Setup the Hamiltonian
def setup_H_full(spin_names,orb_names,comp_H,int_in,mu_in,verbose,mo_den=[]):
    '''
    Add terms to the Hamiltonian depending on the input file.

    Inputs:
    spin_names: List of spins
    orb_names: List of orbitals
    fops: Many-body operators. In order
            (spin_names[0], orb_names[:]), (spin_names[1], orb_names[:]), ...
    comp_H: List of components of the Hamiltonian
    int_in: Input parameters
    verbose: Print out parts of H
    mo_den: Occupation in the band basis

    Outputs:
    ad: Solution to atomic problem
    mu: Chemical potential
    '''

    fops = [(sn,on) for sn, on in product(spin_names,orb_names)]
    print(fops)
    
    # Setup the Hamiltonian
    H=Operator()
    
    # kinetic
    if comp_H['Hkin']:

        H = add_hopping(H,spin_names,orb_names,int_in,verbose=verbose)
        
    # Double counting
    if comp_H['Hdc']:

        if len(int_in['uijkl']) > 1 or len(int_in['tij']) > 1:
            print('WARNING: DC not tested for spin polarized calculations. Good luck!')        
        
        if int_in['dc_typ']==0:
            H = add_double_counting(H,spin_names,orb_names,fops,int_in,mo_den,verbose=verbose)
        elif int_in['dc_typ']==1:
            H = add_hartree_fock_DC(H,spin_names,orb_names,fops,int_in,mu_in['target_occ'])
        elif int_in['dc_typ']==2:
            H = add_cbcn_DFT_DC(H,spin_names,orb_names,fops,int_in,mo_den)
            
    # Interaction
    if comp_H['Hint']:
        H = add_interaction(H,1,spin_names,orb_names,fops,int_in,verbose=verbose)

    # Solve atomic problem
    ad, H, mu = solve_ad_full(H, fops, mu_in)


    return ad, mu
#*************************************************************************************








#*************************************************************************************
# For sorting
def take_second(elem):
    return elem[1]
#*************************************************************************************























#*************************************************************************************
# Electron-hole correlation function
def get_el_h_L_Cyr(T,ad,spin_names,orb_names,eigensys,n_states,w_min,w_max,n_w,eta=0.01):

    '''
    Computes retarded electron-hole correlation function:

    \chi_{abcd}(\omega) = \frac{1}{Z} \sum_{m,n} \frac{e^{-\beta E_n} - e^{-\beta E_m}}{\omega + E_n - E_m + i\eta} \, \langle n | c_a^\dagger c_b | m \rangle \langle m | c_c^\dagger c_d | n \rangle   

    
    Input:
    T: Temperature in K
    ad: atom diag object
    spin_names: List of spins
    orb_names: Orbital names  
    eigensys: Formatted solution to atom diag
    n_lambda: "Unoocupied states"
    n_alpha: "Occupied states," need to replace with Fermi weights
    w_min: Real frequency mesh minimum
    w_max: Real frequency mesh max
    n_w: Real frequency mesh number of points
    eta: imaginary part for broadening
    
    
    Output:
    L2w: Retarded electron-hole correlation function as a triqs GF object
    
    '''
    
    n_orb=len(orb_names)
    n_spin=len(spin_names)
    omegas=np.linspace(w_min,w_max,n_w)

    # Set up Greens function object:
    w_mesh = MeshReFreq(window=(w_min,w_max), n_w=n_w)
    L2w= Gf(mesh=w_mesh, target_shape=[n_spin*n_orb,n_spin*n_orb,n_spin*n_orb,n_spin*n_orb])

    # kB=8.61733262e-5 # eV/K
    beta=1/(T)
    
    den_mats=[]
    Z=0.0
    for alpha in range(min(n_states,len(eigensys))):

        # Accumulate partition function
        Z += np.exp(-beta*eigensys[alpha][1])
        
        for lam in range(min(n_states,len(eigensys))):
            
            if alpha==lam: continue

            # Convert from states in energy order to those in Hilbert space
            n_eig_a=int(eigensys[alpha][2])
            n_eig_l=int(eigensys[lam][2])
        
            # Get desired states in eigenvector basis
            state_eig_a = np.zeros((int(ad.full_hilbert_space_dim)))
            state_eig_a[int(n_eig_a)]=1.0
            state_eig_l = np.zeros((int(ad.full_hilbert_space_dim)))
            state_eig_l[int(n_eig_l)]=1.0
        
            # Construct density matrix
            den_mat=np.zeros((n_spin*n_orb,n_spin*n_orb),dtype='complex128')
            for s1 in range(0,n_spin):
                for s2 in range(0,n_spin):
                    for ii in range(0,n_orb):
                        for jj in range(0,n_orb):
                        
                            den_op=c_dag(spin_names[s1],orb_names[ii]) * c(spin_names[s2],orb_names[jj])

                            xx=ii+s1*n_orb
                            yy=jj+s2*n_orb
                        
                            den_mat[xx,yy]=np.dot(state_eig_a,act(den_op,state_eig_l,ad))

            #den_mat=np.eye(n_spin*n_orb)
            den_mats.append([eigensys[lam][1],eigensys[alpha][1],den_mat,np.einsum('ij,kl->ijkl',den_mat,np.conjugate(den_mat.T))])
            
    L_omega=[]
    for omega in omegas:
        L=0
        for dm in den_mats:
            L+=(1/Z)*1j * (np.exp(-beta*dm[0])-np.exp(-beta*dm[1]))*dm[3] / (omega+1j*eta+dm[0]-dm[1])
            #L+=1j * dm[2] / (omega+dm[0]+1j*eta)

            # TEST
            #for el in np.ndarray.flatten(1j * (1-np.exp(dm[0]/(kB*T)))*dm[2] / (omega+dm[0]+1j*eta)):
            #    if not el==el:
            #        print('NAN!')
            #        print(omega,dm[0],(1-np.exp(dm[0]/(kB*T))),omega+dm[0]+1j*eta)
            #        raise
            
            
            #print((1-np.exp(dm[0]/(kB*T))),dm[0])
            #L+=1j * (1-np.exp(beta*dm[0])) / (omega+dm[0]+1j*eta)
            #print(dm[0])
        L_omega.append(L)


    L_omega=np.stack(L_omega)
        
    for orb_1 in range(n_orb*n_spin):
        for orb_2 in range(n_orb*n_spin):
            for orb_3 in range(n_orb*n_spin):
                for orb_4 in range(n_orb*n_spin):
                    Gw=Gf(mesh=w_mesh, data=L_omega[:,orb_1,orb_2,orb_3,orb_4])
                    L2w[orb_1,orb_2,orb_3,orb_4]<<Gw
        
    return L2w,L_omega, den_mats
        
#*************************************************************************************