import pylab as plt
import os
import sys
from kmeans import kmeans
utility_dir  = os.path.abspath(os.pardir)+'/data/'
sys.path.append(utility_dir)
import load
import utils


def solution_30_31():
    '''
    Usage information:

    Mode 1: To run kmeans for a range of k and plot the error versus k curve,
        set a range using min_k and max_k

    Mode 2: To run kmeans for a certain k and plot the receptor fields, set
        min_k and max_k to a certain k.
    '''


    min_k = 225
    max_k = 225

    print('...starting solution 30,31')

    #loading Cifar as grey scale data, downscaled to 12x12
    size=[12, 12]
    data_dict = load.cifar_grey(size, n_batches=5)
    data = data_dict["data"]

    if(min_k == max_k):
        [D, S, idx, min_err_k] = kmeans(data, k=225)
              # print("Error: " + str(mean_err))
        #for a fixed k, visualize the receptor fields
        utils.show_hidden_layer_rf(X= D.T,
                       save_dir='repflds.png',
                       img_shape=(12, 12),
                       tile_shape=(15, 15),
                       tile_spacing=(1, 1))
    else:
        #to search for the appropriate k plot mean error against k
        min_err_plot=[] #error plot over number of clusters
        for itr_k in range(min_k, max_k):
            [D, S, idx, min_err_k] = kmeans(data, k=itr_k)
              # print("Error: " + str(mean_err))

            min_err_plot.append(min_err_k)
            print(itr_k)

        plt.ion()
        plt.ylabel('mean_error')
        plt.xlabel('number of clusters, k')
        plt.plot(range(min_k, max_k),min_err_plot,'rx')
        plt.show()
        plt.savefig('error_versus_k.png')

    print('finished solution 30,31...')


if __name__ == "__main__":
    print('...starting kmeans')
    solution_30_31()
    print('finished kmeans...')