import numpy as np
import scipy.io
from scipy.special import digamma
import pandas as pd

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import os
import datetime

data_dir = 'data/hw4/'
img_dir = 'img/'

np.random.seed(123)

path = os.path.join(data_dir, 'data.mat')
mat_file = scipy.io.loadmat(path)

beta_matrix = mat_file['beta_matrix']
data = mat_file['data']

M, N = data.shape # M individuals, N genotype loci
_, K = beta_matrix.shape # K ancestor populations

epsilon = 1e-3



def _abs_diff(new, old, eps):
    return all(np.abs(new -old) <= eps)

## VI

def variational_inference(phi_in, gamma_in, beta, N):
    run_info = {}
    run_info['start_time'] = datetime.datetime.now()
    num_iter = 0
    phi = phi_in.copy()
    gamma = gamma_in.copy()
    while True:
        old_phi = phi.copy()
        old_gamma = gamma.copy()
        for n in range(N):
            for i in range(K):
                phi[n,i] += beta[n,i]*np.exp(digamma(gamma[i]))
            phi[n,:] = phi[n,:]/sum(phi[n,:])
        gamma = alpha +phi.sum(axis=0)
        if num_iter % 100 == 0:
            print(num_iter)
        num_iter += 1
        if _abs_diff(gamma, old_gamma, epsilon) and _abs_diff(phi, old_phi, epsilon):
            break
    run_info['num_iter'] = num_iter
    run_info['end_time'] = datetime.datetime.now()
    return phi, gamma, run_info

alphas = [0.01, 0.1, 1, 10]
acc_2 = []
for alpha_const in alphas:
    print(f'alpha: {alpha_const}')
    acc = []
    for indiv in range(M):
        N_i = sum(data[indiv,:] > 0)
        phi = np.ones((N_i,K))*1/K
        alpha = np.ones(K)*alpha_const
        gamma = np.ones(K)*(alpha +N_i/K)
        beta = beta_matrix
        phi_out, gamma_out, run_info = variational_inference(phi, gamma, beta, N_i)
        acc.append([N_i, phi_out, gamma_out, run_info])
    acc_2.append(acc)

theta_list = []
Run_info = []
for row in acc_2:
    N_i_list, phi_list, gamma_list, run_info_list = zip(*row)
    theta_list.append(gamma_list)
    df = pd.DataFrame.from_records(run_info_list)
    df['time'] = (df['end_time'] -df['start_time']).map(lambda td: td.total_seconds())
    Run_info.append(df[['time', 'num_iter']].values)
Theta = np.array(theta_list)
Run_info = np.array(Run_info)

fig, axs = plt.subplots(K, 1, figsize=(24, 8), sharex=True)
for k in range(K):
    ax = axs[k]
    ax.imshow(Theta[k,:,:].T, cmap=plt.cm.Greys, norm=mcolors.Normalize())
    ax.set_title(r'$\alpha$ = {}'.format(alphas[k]))
    ax.yaxis.set_visible(False)
output_path = 'img/hw04_s4_-_vi_lda.png'
plt.suptitle('Individual ancestor population assignments for several hyperpriors\nM={} individuals, K={} ancestor populations'.format(M, K))
plt.savefig(output_path, format='png', bbox_inches='tight')

for k in range(K):
    mtx = Run_info[k,:,:]
    print(k, np.mean(mtx[:,1]))

print(pd.Series(Theta[3,:,:].argmax(axis=1) -Theta[0,:,:].argmax(axis=1)).value_counts())
