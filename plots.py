import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

textsize = 10
fig_width = 6
fig_height = 2

def plot_causal_factorization(SLS_data, F, Dec, Enc):
    # plot sparsity of causal factorization

    epsilon = 1e-10 # threshold to plot sparsity using pyploy.spy

    #F = F[0:-2, 0:-2]
    #Dec = Dec[0:-2, :]
    #Enc = Enc[:, 0:-2]

    (n_rows, n_cols) = np.shape(F)
    band = np.shape(Enc)[0]

    gs2 = gridspec.GridSpec(1, 3, width_ratios = [n_cols, band, n_cols])

    fig = plt.figure()
    axs20 = plt.subplot(gs2[0,0])
    axs21 = plt.subplot(gs2[0,1])
    axs22 = plt.subplot(gs2[0,2])

    axs20.spy(F, epsilon, markersize=0.8, color='r', label='$\mathbf{K}$')
    axs21.spy(Dec, epsilon, markersize=0.8, color='r', label='$\mathbf{D}$')
    axs22.spy(Enc, epsilon, markersize=0.8, color='r', label='$\mathbf{E}$')

    axs20.yaxis.set_major_locator(MaxNLocator(integer=True)) # integer ticks on axis
    axs20.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs21.yaxis.set_major_locator(MaxNLocator(integer=True))
    axs21.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs22.yaxis.set_major_locator(MaxNLocator(integer=True))
    axs22.xaxis.set_major_locator(MaxNLocator(integer=True))

    step_ticks = 10

    axs20.set_xticks(np.arange(step_ticks-1,SLS_data.T*SLS_data.ny,step_ticks), np.arange(step_ticks,SLS_data.T*SLS_data.ny+1, step_ticks))
    axs20.set_yticks(np.arange(step_ticks-1,SLS_data.T*SLS_data.nu,step_ticks), np.arange(step_ticks,SLS_data.T*SLS_data.nu+1, step_ticks))
    axs20.tick_params(axis='both', labelsize=textsize+3)

    axs21.set_xticks(np.arange(step_ticks-1,Dec.shape[1],step_ticks), np.arange(step_ticks,Dec.shape[1]+1, step_ticks))
    axs21.set_yticks(np.arange(step_ticks-1,SLS_data.T*SLS_data.nu,step_ticks), np.arange(step_ticks,SLS_data.T*SLS_data.nu+1, step_ticks))
    axs21.tick_params(axis='both', labelsize=textsize+3)

    axs22.set_yticks(np.arange(step_ticks-1,Enc.shape[0],step_ticks), np.arange(step_ticks,Enc.shape[0]+1, step_ticks))
    axs22.set_xticks(np.arange(step_ticks-1,SLS_data.T*SLS_data.ny,step_ticks), np.arange(step_ticks,SLS_data.T*SLS_data.ny+1, step_ticks))
    axs22.tick_params(axis='both', labelsize=textsize+3)

    axs20.grid(); axs21.grid(); axs22.grid()
    axs20.legend(markerscale=0, handlelength=-0.8); axs21.legend(markerscale=0, handlelength=-0.8); axs22.legend(markerscale=0, handlelength=-0.8)
    fig.set_size_inches(fig_width+2, fig_height+3)
    axs20.set_aspect(1.3); axs21.set_aspect(1.3); axs22.set_aspect(1.3)
    fig.tight_layout()

    return fig

def plot_transmission_times(fig, transmission_times, T, **kwargs):
    plt.step( [0]+transmission_times+[T], [0]+list(range(1,len(transmission_times)+1))+[len(transmission_times)], where='post', **kwargs)
    
    ax=fig.gca()

    ax.set_xlabel("Time step")
    ax.set_ylabel("Nbr. of transmissions")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # integer ticks on axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid()
    #fig.tight_layout()

    plt.rc('xtick', labelsize=textsize) 
    plt.rc('ytick', labelsize=textsize) 
    plt.rcParams.update({'font.size': textsize})
    fig.set_size_inches(fig_width, fig_height)

def plot_transmissions_vs_tol(list_truncation_tol, list_n_transmission, **kwargs):
    fig = plt.figure()
    plt.plot(list_truncation_tol, list_n_transmission, **kwargs)
    plt.xscale("log")

    ax=fig.gca()
    ax.set_xlabel('$\epsilon$')
    ax.set_ylabel('Nbr. of transmissions')
    ax.grid()

    ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # integer ticks on axis

    plt.rc('xtick', labelsize=textsize) 
    plt.rc('ytick', labelsize=textsize) 
    plt.rcParams.update({'font.size': textsize})
    fig.set_size_inches(fig_width, fig_height)

    return fig

def plot_transmissions_vs_gamma(fig, list_gamma, list_n_transmission_gamma, **kwargs):
    plt.plot(list_gamma, list_n_transmission_gamma, **kwargs)

    ax=fig.gca()
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel('Nbr. of transmissions')

    ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # integer ticks on axis
    ax.legend()
    

    plt.rc('xtick', labelsize=textsize) 
    plt.rc('ytick', labelsize=textsize) 
    plt.rcParams.update({'font.size': textsize})
    fig.set_size_inches(fig_width, fig_height)
    ax.grid(linestyle='-')