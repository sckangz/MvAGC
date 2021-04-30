#!usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import time
import random
# import tensorflow as tf
import numpy as np
import scipy
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from time import *
import warnings
warnings.filterwarnings("ignore")
# warnings.simplefilter('error',ComplexWarning)


def read_data(str):
    data = sio.loadmat('{}.mat'.format(str))
    if(str == 'large_cora'):
        X = data['X']
        A = data['G']
        gnd = data['labels']
        gnd = gnd[0, :]
    else:
        X = data['feature']
        A = data['MAM']
        B = data['MDM']
        av=[]
        av.append(A)
        av.append(B)
        gnd = data['label']
        gnd = gnd.T
        gnd = np.argmax(gnd, axis=0)

    return X, av, gnd


def FGC_cora_modified(X, av, gnd, a, k, ind):
        # Store some variables
    gama=-1
    final=[]
    nada = [1, 1]
    X_hat_list=[]
    X_hat_anchor_list=[]
    A_hat_list=[]
    for i in range(2):
        A=av[i]
        N = X.shape[0]
        # print("N = {}".format(N))
        Im = np.eye(len(ind))
        In = np.eye(N)
        if sp.issparse(X):
            X = X.todense()

        # Normalize A
        A = A + In
        D = np.sum(A, axis=1)
        D = np.power(D, -0.5)
        D[np.isinf(D)] = 0
        D = np.diagflat(D)
        A = D.dot(A).dot(D)

        # Get filter G
        Ls = In - A
        G = In - 0.5 * Ls
        G_ = In
        X_hat = X
        for i in range(k):
            # G_ = G_.dot(G)
            X_hat = G.dot(X_hat)
        X_hat_list.append(X_hat)
        A_hat = (A)[ind]  # (m,n)
        A_hat_list.append(A_hat)
        X_hat_anchor_list.append(X_hat[ind])
    begin_time = time()
    # Set the order of filter
    for t in range(5):
        tmp1=0
        tmp2=0
        for i in range(2):
            tmp1 =tmp1+nada[i]*(X_hat_anchor_list[i].dot(X_hat_anchor_list[i].T) + a * Im)
        for i in range(2):
            tmp2 = tmp2+nada[i]*(X_hat_anchor_list[i].dot(X_hat_list[i].T) + a * A_hat_list[i])
        S = np.linalg.inv(tmp1).dot(tmp2)
        for i in range(2):
            nada[i] = (-((np.linalg.norm(X_hat_list[i].T - (X_hat_anchor_list[i].T).dot(S))) ** 2 + a * (np.linalg.norm(S - A_hat_list[i])) ** 2) / (gama)) ** (1 / (gama - 1))
            print("nada值")
            print(nada[i])
    #     res = 0
    #     for j in range(2):
    #         res = res + nada[j] * ((np.linalg.norm(X_hat_list[i].T - (X_hat_anchor_list[i].T).dot(S))) ** 2 + a * (
    #             np.linalg.norm(S - A_hat_list[i])) ** 2) + (nada[j]) ** (gama)
    #     final.append(res)
    #     print(res)
    # sio.savemat("a.mat", {'res': final})
    return S, begin_time


def main(X, av, gnd, m, a, k, ind):

    N = X.shape[0]
    begin_time_filter = time()
    types = len(np.unique(gnd))
    S, begin_time = FGC_cora_modified(X, av, gnd, a, k, ind)
    D = np.sum(S, axis=1)
    D = np.power(D, -0.5)
    D[np.isinf(D)] = 0
    D[np.isnan(D)] = 0
    D = np.diagflat(D)  # (m,m)

    S_hat = D.dot(S)  # (m,n)

    S_hat_tmp = S_hat.dot(S_hat.T)  # (m,m)
    S_hat_tmp[np.isinf(S_hat_tmp)] = 0
    S_hat_tmp[np.isnan(S_hat_tmp)] = 0
    # sigma, E = scipy.linalg.eig(S_hat_tmp)
    E, sigma, v = sp.linalg.svds(S_hat_tmp, k=types, which='LM')
    sigma = sigma.T
    sigma = np.power(sigma, -0.5)
    sigma[np.isinf(sigma)] = 0
    sigma[np.isnan(sigma)] = 0
    sigma = np.diagflat(sigma)
    C_hat = (sigma.dot(E.T)).dot(S_hat)
    C_hat[np.isinf(C_hat)] = 0
    C_hat[np.isnan(C_hat)] = 0
    C_hat = C_hat.astype(float)
    kmeans = KMeans(n_clusters=types, random_state=37).fit(C_hat.T)

    predict_labels = kmeans.predict(C_hat.T)

    cm = clustering_metrics(gnd, predict_labels)
    ac, nm, f1,adj = cm.evaluationClusterModelFromLabel(m,a,k)
    end_time = time()
    tot_time = end_time - begin_time
    tot_time_filter = end_time - begin_time_filter
    return ac, nm, f1,adj, tot_time, tot_time_filter


def lower_bound(p, rd):
    l = 0
    r = len(p) - 1
    while(l < r):
        # print("rd = {}, l = {}, r= {}".format(rd, l, r))
        mid = (l + r) // 2
        if(p[mid] > rd):
            r = mid
        else:
            l = mid + 1
    # print("rd = {}, l = {}, r= {}".format(rd, l, r))
    return l


def node_sampling(A, m, alpha):
    D = np.sum(A[0], axis=1).flatten()+np.sum(A[1], axis=1).flatten()

    if(len(np.shape(D)) > 1):
        D = D.A[0]
        print(1)

    D = D**alpha
    D=D/10000
    print(D)
    tot = np.sum(D)
    print(tot)
    p = D / tot
    print(p)
    for i in range(len(p) - 1):
        p[i + 1] = p[i + 1] + p[i]
    print(p)
    ind = []
    vis = [0] * len(D)
    while(m):
        while(1):
            rd = np.random.rand()
            pos = lower_bound(p, rd)
            if(vis[pos] == 1):
                continue
            else:
                vis[pos] = 1
                ind.append(pos)
                m = m - 1
                break
    return ind


def func(X, A, gnd):
    m_init_list = [80] #anchor numbers
    a_list = [100] #second term
    k_init_list = [2] #juanjijieshu
    f_alpha_init_list = [3] #important node
    k_list = []
    aa_list = []
    i_list = []
    ac_list = []
    nm_list = []
    f1_list = []
    adj_list=[]
    tm_list = []
    tm_list_filter = []
    f_alpha_list = []

    N = X.shape[0]
    tot_test = 1
    ac_max = 0.0
    xia = 0
    tot = 0

    # print(node_sampling(A, 20))
    for k in k_init_list:
        for i in m_init_list:
            # print("now k = {}, now m = {}".format(k, i))
            for alpha in f_alpha_init_list:
                ind = node_sampling(A, i, alpha)
                ac_mean = 0
                nm_mean = 0
                f1_mean = 0
                adj_mean=0
                tm_mean = 0
                for a in a_list:
                    # continue
                    acc, nmm, f11,adj, tm, tm_filter = main(
                        X, A, gnd, i, a, k, ind)
                    print("m = {},k = {}, f_alpha = {},a  ={}, ac = {}, nmi = {}, f1 = {},adj={}, tm = {}, tm_filter = {}".format(
                        i, k, alpha, a, acc, nmm, f11,adj, tm, tm_filter))
                    if(ac_mean < acc):
                        ac_mean = acc
                        nm_mean = nmm
                        f1_mean = f11
                        adj_mean=adj
                        tm_mean = tm
                        tm_mean_filter = tm_filter
                    i_list.append(i)
                    k_list.append(k)
                    aa_list.append(a)
                    f_alpha_list.append(alpha)
                    ac_list.append(ac_mean)
                    nm_list.append(nm_mean)
                    f1_list.append(f1_mean)
                    adj_list.append(adj_mean)
                    tm_list.append(tm_mean)
                    tm_list_filter.append(tm_mean_filter)
                print("m = {}, k ={},f_alpha = {}, ac_mean = {},nm_mean = {},f1_mean = {},adj_mean={},tm_mean = {},tm_mean_filter = {}\n".format(
                    i, k, alpha, ac_mean, nm_mean, f1_mean,adj_mean, tm_mean, tm_mean_filter))

                if(ac_mean > ac_max):
                    xia = tot
                    ac_max = ac_mean

                tot += 1

    for i in range(len(i_list)):
        print("m = {},k = {},f_alpha = {}, ac_mean = {}, nm_mean = {}, f1_mean = {},adj_mean={},tm_mean = {},tm_mean_filter ={}".format(
            i_list[i], k_list[i], f_alpha_list[i], ac_list[i], nm_list[i], f1_list[i],adj_list[i], tm_list[i], tm_list_filter[i]))
    print("the best result is ")
    print("m = {},k = {},f_alpha = {}, ac_mean = {}, nm_mean = {}, f1_mean = {},adj_mean={},tm_mean = {},tm_mean_filter = {}".format(
        i_list[xia], k_list[xia], f_alpha_list[xia], ac_list[xia], nm_list[xia], f1_list[xia],adj_list[xia], tm_list[xia], tm_list_filter[xia]))
    return i_list[xia], k_list[xia], f_alpha_list[xia], ac_list[xia], nm_list[xia], f1_list[xia],adj_list[xia], tm_list[xia], tm_list_filter[xia]


if __name__ == '__main__':
    dataset = 'imdb5k'
    X, A, gnd = read_data(dataset)
    # number of epoch
    tt = 1
    m_best_list = []
    k_best_list = []
    f_alpha_best_list = []
    ac_best_list = []
    nm_best_list = []
    f1_best_list = []
    adj_best_list = []
    tm_best_list = []
    tm_filter_best_list = []
    for i in range(tt):
        nowm, nowk, nowf, nowac, nownm, nowf1,nowadj, nowtm, nowtmf = func(X, A, gnd)
        m_best_list.append(nowm)
        k_best_list.append(nowk)
        f_alpha_best_list.append(nowf)
        ac_best_list.append(nowac)
        nm_best_list.append(nownm)
        f1_best_list.append(nowf1)
        adj_best_list.append(nowadj)
        tm_best_list.append(nowtm)
        tm_filter_best_list.append(nowtmf)
        print("iteration {}, m = {}, k = {}, f_alpha = {}, ac = {}, nm = {}, f1 = {},adj={}, tm = {}, tm_filter = {}".format(
            i + 1, nowm, nowk, nowf, nowac, nownm, nowf1,nowadj, nowtm, nowtmf))
    for i in range(len(ac_best_list)):
        print("iteration {}, m = {}, k = {}, f_alpha = {}, ac = {}, nm = {}, f1 = {},adj={}, tm = {}, tm_filter = {}".format(
            i + 1, m_best_list[i], k_best_list[i], f_alpha_best_list[i], ac_best_list[i], nm_best_list[i], f1_best_list[i],adj_best_list[i], tm_best_list[i], tm_filter_best_list[i]))
    print("ac_mean = {}, ac_std = {}".format(
        np.mean(ac_best_list), np.std(ac_best_list, ddof=1)))
    print("nm_mean = {}, nm_std = {}".format(
        np.mean(nm_best_list), np.std(nm_best_list, ddof=1)))
    print("f1_mean = {}, f1_std = {}".format(
        np.mean(f1_best_list), np.std(f1_best_list, ddof=1)))
    print("adj_mean = {}, adj_std = {}".format(
        np.mean(adj_best_list), np.std(adj_best_list, ddof=1)))
    print("tm_mean = {}, tm_std = {}".format(
        np.mean(tm_best_list), np.std(tm_best_list, ddof=1)))
    print("tmf_mean = {}, tmf_std = {}".format(
        np.mean(tm_filter_best_list), np.std(tm_filter_best_list, ddof=1)))
