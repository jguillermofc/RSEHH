"""
Save approximation set.
"""

import numpy as np
import matplotlib.pyplot as plt
import hashlib
import os



def saveApproximationSet(A, ppf, algorithm, nobj, problem, ss_size, run, seq_file, mode='save_all'):
    """Draws and saves a given approximation set"""
    N,m = np.shape(A)   
    name, _ = os.path.splitext(seq_file)
    md5_hash = hashlib.md5(name.encode()).hexdigest()
    DIR =  f"Results/Approximations/{ppf}/{nobj:02d}D/{problem}"

    fname_prefix = algorithm+'_'+problem+'_ss{0:0=d}_{1:0=2d}D'.format(ss_size, m)+'_'+md5_hash+'_R{0:0=2d}'.format(run)
    os.makedirs(DIR, exist_ok=True)      
    
    if mode == 'save_txt':
        fname_pof = os.path.join(DIR, fname_prefix + '.pof')
        np.savetxt(fname_pof, A, fmt='%.6e', header=str(N)+' '+str(m))
    else:
        if m == 2:
            plt.scatter(A[:,0], A[:,1], color=(0.7, 0.7, 0.7), edgecolors=(0.4, 0.4, 0.4))
        elif m == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(30,45)
            ax.scatter(A[:,0], A[:,1], A[:,2], color=(0.7, 0.7, 0.7), edgecolors=(0.4, 0.4, 0.4), alpha=1)
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        else:
            for i in range(0,N):
                plt.plot(A[i], color=(0.5, 0.5, 0.5))
            plt.xlabel('Objective function',rotation=0)
            plt.ylabel('Objective value')
            x = []
            labels = []
            for i in range(0,m):
                x.append(i)
                labels.append(str(i+1))
            plt.xticks(x,labels)
        plt.title(algorithm+' on '+problem)
        plt.tight_layout()
        if mode == 'save_all':
            fname_pof = os.path.join(DIR, fname_prefix + '.pof')
            fname_png = os.path.join(DIR, fname_prefix + '.png')
            np.savetxt(fname_pof, A, fmt='%.6e', header=str(N)+' '+str(m))
            plt.savefig(fname_png)
            plt.close()
        elif mode == 'save_fig':
            fname_png = os.path.join(DIR, fname_prefix + '.png')
            plt.savefig(fname_png)
            plt.close()
        elif mode == 'plot':
            plt.show()
            plt.close()
    return
