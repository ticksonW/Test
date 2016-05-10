import pylab as plt

import os
import sys
utility_dir  = os.path.abspath(os.pardir)+'/data/'
sys.path.append(utility_dir)


import load
import utils
from kmeans import kmeans

def problem_30_31():

    #loading Cifar as grey scale data, downscaled to 12x12
    size=[12, 12]
    data_dict = load.cifar_grey(size, n_batches=1)
    data = data_dict["data"]

    # data = np.loadtxt('testInput21A.txt',delimiter=',', dtype=float,
     # skiprows=1)
    # data = data[:5,:]
    # # ZCA-whitening

    min_err_plot=[] #error plot over number of clusters

    min_k = 100
    max_k = 100
    for itr_k in range(min_k, max_k):
        [D, S, idx, min_err_k] = kmeans(data, k=itr_k)
          # print("Error: " + str(mean_err))

        min_err_plot.append(min_err_k)
        print(itr_k)
        # if (min_err_k < min_err):
        #     min_err = min_err_k
        #     D_optimal = D
        #     S_optimal = S
        #     cluster_idx_opt = idx
        #     print("Updated")



    plt.ion()
    plt.ylabel('mean_error')
    plt.xlabel('number of clusters, k')
    plt.plot(range(min_k, max_k),min_err_plot,'rx')
    plt.show()
    plt.savefig('kmeans.png')
    print('done')

def visualize_repfld():
    size=[12, 12]
    data_dict = load.cifar_grey(size, n_batches=1)
    data = data_dict["data"]
    [D, S, idx, min_err_k] = kmeans(data, k=100)

    utils.show_hidden_layer_rf(X= D.T,
                               save_dir='100_repflds2.png',
                               img_shape=(12, 12),
                               tile_shape=(10, 10),
                               tile_spacing=(1, 1))

    utils.plot_rpfld(input = D.T,
                     tile_shape= (10,10),
                     img_shape=(12,12),
                     save_dir= '100_repflds.png')



if __name__ == "__main__":
    visualize_repfld()
    problem_30_31()
    print('finished')