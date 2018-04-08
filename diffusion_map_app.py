import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.spatial import ConvexHull
import sys




def calculate_dist_matrix(t):
    #aij = half dist squared
    #t = md.load('output2.dcd', top = 'ala-dipeptide.pdb')
    t.center_coordinates()
    t.superpose(t)
    a = np.zeros((t.n_frames, t.n_frames))
    print('calculating dist matrix s')
    sys.stdout.write("calculating dist matrix : ")
    sys.stdout.flush()
    for i in range(np.shape(a)[0]):
        #print('done',i)
        msg = "item %i of %i" % (i, t.n_frames)
        sys.stdout.write(msg + chr(8) * len(msg))
        sys.stdout.flush()
        for j in range(i+1):
            a[i][j] = md.rmsd(t[i], t[j]) ######## ||t1 - t2||
            a[j][i] = a[i][j]
    sys.stdout.write(str(t.n_frames)+ "  DONE" + " "*len(msg)+"\n")
    sys.stdout.flush()
    return a # aij*episilon

def compute_epsilon(dist_matrix,sample):
    a = -(dist_matrix**2)/2
    #sample = [-5,-4,-3,-2,-1,0,1,2,3,4]
    sys.stdout.write("calculating sigma ij aij vs epsilon : ")
    sys.stdout.flush()
    l = np.zeros(len(sample))
    logEpsilon = np.zeros(len(sample))
    for i in range(len(sample)):
        msg = "item %i of %i" % (i, len(sample))
        sys.stdout.write(msg + chr(8) * len(msg))
        sys.stdout.flush()
        e = 10**sample[i]
        logEpsilon[i] =sample[i]
        l[i] = np.log10(np.sum(np.exp(a/e)))
    sys.stdout.write(str(len(sample))+ "  DONE" + " "*len(msg)+"\n" + "ploting\n")
    sys.stdout.flush()
    logl = l
    plt.plot(logEpsilon,logl,'ro')
    plt.plot([-7,5],[logl.max(),logl.max()],'g')
    plt.plot([-7,5],[logl.min(),logl.min()],'g')
    mean = (logl.max()+logl.min())/2
    plt.plot([-7,5],[mean,mean],'g')
    plt.ylabel('log10(L) ')
    plt.xlabel('log10(E)')
    plt.grid(True)
    plt.show()
    """
    plt.plot(logEpsilon,l,'ro')
    plt.ylabel('log10(L) ')


    plt.grid(True)
    plt.show()
    """
    return logl, logEpsilon

def diffusion_map_alpha_half(dist_matrix,epsilon):

    a = np.exp(-(dist_matrix**2)/(2*epsilon))
    w = np.zeros(np.shape(a))
    a_row_sum = np.sqrt(a.sum(axis=1))
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[0]):        #wij = wji doooooooooooo   a is symmetric
            w[i][j]=(a[i][j])/(a_row_sum[i]*a_row_sum[j])   #alpha = 0.5 sigma i aij = sigma j aij ^
    dinvrse = np.diag(1/w.sum(axis = 0))
    p = np.matmul(dinvrse, w)
    return p,a_row_sum

def diffusion_map_generic(a,epsilon,alpha = 0.5):
    a = np.exp(-(dist_matrix**2)/(2*epsilon))
    w = np.zeros(np.shape(a))
    a_row_sum = np.power(a.sum(axis=1),alpha)
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[0]):        #wij = wji doooooooooooo   a is symmetric
            w[i][j]=(a[i][j])/(a_row_sum[i]*a_row_sum[j])   #alpha = 0.5 sigma i aij = sigma j aij ^
    dinvrse = np.diag(1/w.sum(axis = 0))
    p = np.matmul(dinvrse, w)
    return p, a.sum(axis=1)

def ramachandran_plot(t):
    l,phi = md.compute_phi(t)
    n,psi = md.compute_psi(t)
    phi = np.rad2deg(phi)
    psi = np.rad2deg(psi)
    plt.plot(phi,psi,'go',markersize = 1)
    plt.axis([-180, 180, -180, 180])
    plt.grid(True)  
    plt.show()  
    return phi, psi

def nystorm(samp_traj, newframe, epsilon, a_row_sum, eigen_values, eigen_vectors, dim):
    new_a_row = np.exp(-np.square(md.rmsd(samp_traj,newframe))/(2*epsilon))  # previously aligned
    new_w_row = np.zeros(np.shape(new_a_row))
    a_row_sum = np.sqrt(a_row_sum)
    for i in range(len(new_a_row)):
        new_w_row[i] = new_a_row/a_row_sum[i]
    new_p = new_w_row/np.sum(new_w_row)
    nystrom = np.zeros( 1+dim, )
    for i in range(1+dim):
        nystorm[i] = np.dot(new_p,eigen_vectors[i]/eigen_values[i])
    return nystorm













"""
t = md.load('output2.dcd', top = 'ala-dipeptide.pdb')
t = t.removesolvent()
dist_matrix = calculate_dist_matrix(t)
sample = [-6,-7,-8,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]
sample = np.append(sample,np.arange(-2,2,0.1))
logl, logEpsilon = compute_epsilon(a,sample)
plt.plot(logEpsilon,logl,'ro')
plt.plot([-5,5],[logl.max(),logl.max()],'g')
plt.plot([-5,5],[logl.min(),logl.min()],'g')
mean = (logl.max()+logl.min())/2
plt.plot([-5,5],[mean,mean],'g')
plt.ylabel('log10(L) ')
plt.xlabel('log10(E)')
plt.grid(True)
plt.show()

#epsilon = 10**-2.7635  ############ TBD
#   epsilon = 10**   -2.7648
p,a_row_sum = diffusion_map_alpha_half(a,epsilon)
eigen_values, eigen_vectors = LA.eig(p)
savefolder = 'abc/'   ########### TBD
np.save(savefolder+'aij',np.exp(a/epsilon))
np.save(savefolder+'eigen_values',eigen_values)
np.save(savefolder+ 'eigen_vectors',eigen_vectors)
np.save(savefolder+ 'p',p)

"""""






