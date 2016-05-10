import numpy as np
import theano as th
import theano.tensor as T

import os
import sys
utility_dir  = os.path.abspath(os.pardir)+'/data/'
sys.path.append(utility_dir)


def ZCA(data, n_component=2):

    # data standardization
    x = T.matrix('x')
    eps = T.scalar('eps')
    y = (x - T.mean(x, axis=0)) / T.sqrt(T.var(x) + eps)
    standardize = th.function([x, eps], y)
    #s_data = standardize(data, 10)

    # zca whitening
    x_n = T.matrix('x_n')  # normalized input
    eps2 = T.scalar('eps2')  # small esp to prevent div by zero
    #x_c = x_n - T.mean(x_n, 0)  # centered input
    x_cov = T.dot(x_n.T, x_n) / x_n.shape[0]  # variance of input
    u, s, v = T.nlinalg.svd(x_cov)

    # get_s = th.function([x_n], s)
    # sigma = get_s(n_data)

    z = T.dot(T.dot(u, T.nlinalg.diag(1. / T.sqrt(s + eps2))), u.T)
    x_zca = T.dot(x_n, z.T[:, :n_component])
    zca_whiten = th.function([x_n, eps2], x_zca)
    return  zca_whiten(standardize(data, 0.1), 0.01)



def kmeans(data, k):

    n = data.shape[1] #n-dimension
    m = data.shape[0] #n-data

    D_optimal = np.random.rand(n, k)
    S_optimal = np.zeros((k,m), dtype=th.config.floatX)

    #The literature uses the transpose version
    X_ZCA = ZCA(data, n).T

    X = th.shared(
        value=X_ZCA,
        name='D',
        borrow=True
    )

    # centroid
    D = th.shared(
        value=np.random.rand(n, k)*4-2,
        name='D',
        borrow=True
    )

    # selection matrix
    S = th.shared(
        value=np.zeros((k,m), dtype=th.config.floatX),
        name='S',
        borrow=True
    )

    S_zero = th.shared(
        value=np.zeros((k,m), dtype=th.config.floatX),
        name='S_zero',
        borrow=True
    )

    # dim(Y)=R^{k,m} along each column gives the score of one
    # data point for each cluster
    Y = T.dot(D.T, X)

    # argmax over rows = argmax over cluster this is the best cluster
    # index for each data points
    I = T.argmax(abs(Y), axis=0) #best cluster index
    # update S: centroid selection
    # each column is a vector for each data point which tells
    # has a 1 at its corresponding centroid, otherwise 0
    S_new = T.set_subtensor(S_zero[I, T.arange(m)], Y[I, T.arange(m)])

    D_t = (T.dot(X, S.T) + D)
    D_new = (D_t) / T.sqrt(T.sum(T.square(D_t),axis=0))

    update_s = th.function([],S_new)
    update_d = th.function([],D_new)

    # error function
    Error = T.mean(T.sqrt(T.sum(T.square(T.dot(D,S)-X), axis=0)))

    train_model = th.function(
        inputs=[],
        outputs=[Error, I]
    )

    n_epoch = 15
    n_init = 10

    min_err = np.infty

    for ni in range(n_init):
        D.set_value((np.random.rand(n, k)*4-2) ,borrow=True)
        for epoch in range(n_epoch):

            S.set_value(update_s() ,borrow=True)
            D.set_value(update_d() ,borrow=True)

            mean_err, cluster_idx = train_model()
            # if(epoch == n_epoch-1):
            #     print("Init: " + str(ni) + ", Mean Error: "+str(mean_err))
            # print("Error: " + str(mean_err))

        if (mean_err < min_err):
            min_err = mean_err
            D_optimal = D.get_value(borrow= True)
            S_optimal = S.get_value(borrow= True)
            cluster_idx_opt = cluster_idx
            #print("Updated")

    # plt.ion()
    # plt.clf()
    # plt.scatter(X_ZCA[0,:], X_ZCA[1,:], 50, cluster_idx_opt)
    # plt.show()
    # plt.savefig("kmeans.png")
    return [D_optimal, S_optimal, cluster_idx_opt, min_err]

