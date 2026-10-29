#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used for the analysis in [Title]

This document contains all functions developed for this study or adapted from external sources.
External function are accordingly identified in their docstrings.

Note that this code provides functions only, for the workflow see the accompanying notebooks.
"""

#### IMPORTS

import pandas as pd 
import numpy as np
from numpy import zeros, sqrt, fabs, cos, sin, tile, vstack, empty, inf, mean, arange
from numpy.linalg import svd
from numpy import (
    zeros,
    sqrt,
    fabs,
    cos,
    sin,
    tile,
    vstack,
    empty,
    cov,
    inf,
    mean,
    arange,
)
from numpy.random import randn
from scipy.linalg import norm, solve
from statistics import mode

import os
from copy import deepcopy
from joblib import Parallel, delayed
import tqdm

from PIL import Image
from skimage import measure, io
from skimage.filters import gaussian, threshold_otsu
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from sklearn.metrics import f1_score

import collections
import math
import random
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import time

import fdasrsf.geodesic as gd
import fdasrsf.curve_stats as cs
import fdasrsf.curve_functions as cf
import fdasrsf.utility_functions as uf

# ================
#### NEW FUNCTIONS
# ================

#### Object identification

def best_mask_contour(c_bm):
    cont = measure.find_contours(c_bm.astype(np.float64), 0.5)
    outline = cont[0]
    return outline[:,1], outline[:,0]

def get_mask(
        img,
        checkpoint_path,
        b_box,
        model_type= "vit_h"
):
    #load model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to("cpu")
    
    # convert image
    img = io.imread("temp.png", as_gray=False)
    if img.max() <= 1.0:
        img = (img*255).astype(np.uint8)
    if img.shape[2] == 4: 
        img = img[:, :, :3]
        
    # predictor set up
    predictor = SamPredictor(sam)
    predictor.set_image(img)
        
    # predict all possible masks and choose the best one
    masks, scores, _ = predictor.predict(box=b_box, multimask_output=True) 
    bm_index = np.argmax(scores)
    bm = masks[bm_index]    
    
    # save Mask in variable
    return bm

def bb_contour(file_path, checkpoint_path):
    # load image
    img = load_resize_image(file_path, resize=0.75)   
    # get contour
    x1, y1 = extract_contour(img)
    # in case no contour can be identified
    if len(x1) == 0: 
        print("no shape identified")
        return None, None, None
    
    # create the bounding box
    leeway=15
    min_x = int(max(min(x1) - leeway, 0))
    max_x = int(min(max(x1) + leeway, np.shape(img)[1]))
    min_y = int(max(min(y1) - leeway, 0))
    max_y = int(min(max(y1) + leeway, np.shape(img)[0]))
    b_box = np.array([min_x, min_y, max_x, max_y])
    
    # crop image & convert to RGB
    cropped_img = img[min_y:max_y, min_x:max_x]
    cropped_img = np.stack([cropped_img, cropped_img, cropped_img], axis = -1)
    
    # generate mask with SAM
    bm = get_mask(img, checkpoint_path, b_box)
    
    # crop mask to fit image
    c_bm = bm[min_y:max_y, min_x:max_x]
    # find a contour based on the best mask    
    x,y = best_mask_contour(c_bm)
    if x is None or len(x) == 0:
        return None, None, None
    
    # reparameterize
    xs, ys = reparam(x, y, 100)
    
    return xs, ys, cropped_img

#### PCA Functions

def pca_pd(pca, n_points, n_dims = 2, n_pcs = 3):
    """
  Reconstruct principal deformation shapes along PCA axes.

  For each principal component, 10 shapes are reconstructed at
  positions from -2 to +2.5 standard deviations in steps of 0.5 SD.

  Parameters
  ----------
  pca : fitted PCA object
      Fitted PCA model providing eigenvalues and inverse_transform()from the prince module.
  n_points : int
      Number of coordinate points per shape.
  n_dims : int, default=2
      Number of spatial dimensions per point.
  n_pcs : int, default=3
      Number of principal components for which PDs are reconstructed.

  Returns
  -------
  pandas.DataFrame
      Reconstructed coordinates for 10 principal deformation shapes
      per principal component.
  """
    pd_dict = {}
    ev = np.asarray(pca.eigenvalues_)
    
    for pc in range(n_pcs):
        for i in range(1,11):
            # Displacement based on shape_pca:
            displacement = (0.5*(i-5)*np.sqrt(ev[pc]))
            
            # Position in PCA space
            pc_scores = pd.DataFrame(
                np.zeros((1, pca.n_components)),
                columns=range(pca.n_components))
            
            # Vary only the current PC
            pc_scores.iloc[0, pc] = displacement
            
            # Transcorm PCA-coords back to shape coords
            shape = (pca.inverse_transform(pc_scores)
                     .to_numpy()
                     .reshape(n_points, n_dims))
            
            pd_dict[f'PD_{pc+1}_{i}_x']=shape[:, 0]
            pd_dict[f'PD_{pc+1}_{i}_y']=shape[:, 1]
            
    return pd.DataFrame(pd_dict)

#=====================================================
#### Functions copied or adapted from external sources
#=====================================================
"""
The following functions were provided for the paper 'Classifying Organisms and Artefacts by Their Shapes
' by Salili-James et al. 2022

The original source code is distributed under the GNU General Public License v3.0.

DOI: https://doi.org/10.1098/rsif.2022.0493
Original Repository: https://github.com/smarsland/ClassifyingOrganismsArtefactsShape/tree/main
"""

#### Contour extraction Functions

def load_resize_image(
    imgpath,
    resize=1,
    resizeExtraScale1=0.8,
    resizeExtraScale2=0.55,
    large=1000,
    extraLarge=3000,
):
    # Load original image.
    foo = Image.open((imgpath))
    p = foo.size
    if p[0] > extraLarge:
        resize = resize * resizeExtraScale2
    elif p[0] > large:
        resize = resize * resizeExtraScale1
    # Resize image.
    # This saves on computation cost. If you don't wish to resize, then set resize=1.
    foo = foo.resize((int(p[0] * resize), int(p[1] * resize)), Image.LANCZOS)
    # Temporarily save the resized image.
    foo.save("temp.png", quality=100)
    foo.close()
    # Load resized image.
    img = io.imread("temp.png", as_gray=True)
    return img

def extract_contour(img):
    # 1) Threshold with Otsu
    img_gaussian = gaussian(img, 5)
    thresh = threshold_otsu(img_gaussian)
    binary = img_gaussian > thresh
    # 2) Find all contours:
    contours = measure.find_contours(binary, 0.8)
    # 3) Get longest contour (outline)
    k = np.argmax([len(c[:,0]) for c in contours])
    outline = contours[k]
    return outline[:,1],outline[:,0]

def reparam(x, y, npoints):
    # This function reparametrizes the curve to have npoints.

    tst = np.zeros((len(x), 2))
    tst[:, 0] = x
    tst[:, 1] = y

    p = tst
    dp = np.diff(p, axis=0)
    pts = np.zeros(len(dp) + 1)
    pts[1:] = np.cumsum(np.sqrt(np.sum(dp * dp, axis=1)))
    newpts = np.linspace(0, pts[-1], int(npoints))
    newx = np.interp(newpts, pts, p[:, 0])
    newy = np.interp(newpts, pts, p[:, 1])

    return newx, newy


#### Karcher Mean / tPCA Functions

def karcher_cov(v):
    """
    This calculates the mean of a set of curves

    """
    M, N, K = v.shape
    tmpv = np.zeros((M * N, K))
    for i in range(0, K):
        tmp = v[:, :, i]
        tmpv[:, i] = tmp.flatten()

    return np.cov(tmpv)

def karcher_calc(beta, q, betamean, mu, basis, closed, method):
    # Compute shooting vector from mu to q_i
    w, d = cf.inverse_exp_coord(betamean, beta, closed, method)

    # Project to tangent space of manifold to obtain v_i
    if closed == 0:
        v = w
    else:
        v = cf.project_tangent(w, q, basis)

    return (v, d)

def align(
    sample,
    open_closed="C",
    PC=3,
    scaleCurve=False,
    returnDF=1,
    maxVar=10,
    pd_plots=False,
):
    beta = deepcopy(sample)
    obj_c = fdacurve(beta, mode=open_closed, scale=scaleCurve)
    obj_c.srvf_align()
    obj_c.karcher_mean
    q_mean = obj_c.q_mean
    mn = obj_c.beta_mean
    aligned = obj_c.betan
    cov = karcher_cov(obj_c.v)
    coef, pd_results = shape_pca(
        obj_c.v, cov, obj_c.beta, q_mean, no=PC, pd_plots=pd_plots
    )
    _, s, _ = svd(cov)
    var = np.cumsum(s) / np.sum(s) * 100
    var = var[: max([maxVar, PC])]
    
    pct_var = 100 * s / np.sum(s)
    pct_var_cum = np.cumsum(pct_var)
    
    var_df = pd.DataFrame({
        'PC': [f"PC{i}" for i in range (1, PC+1)],
        'eigenvalue': s[:PC],
        '%variance': pct_var[:PC],
        '%variance_cum': pct_var_cum[:PC],
        }) 
    mnx, mny = rescale(mn[0], mn[1])
    mn[0] = mnx
    mn[1] = mny
    if returnDF == 1:
        tpca = pd.DataFrame(coef).T
        if PC == 3:
            tpca = tpca.rename(columns={0: "PC1", 1: "PC2", 2: "PC3"})
        karchermean = pd.DataFrame([mn[0], mn[1]]).T
        karchermean = karchermean.rename(columns={0: "x", 1: "y"})
    else:
        tpca = deepcopy(coef)
        karchermean = deepcopy(mn)
    if pd_plots:
        return aligned, karchermean, tpca, var, var_df, pd_results
    else:
        return aligned, karchermean, tpca, var, var_df

def shape_pca(v, C, beta, q_mean, no=3, pd_plots=False):
    """
    Computes principal direction of variation specified by no. N is
    Number of shapes away from mean. Creates 2*N+1 shape sequence

    :param no: number of direction (default 3)
    """
    U1, s, V = svd(C)
    U = U1[:, 0:no]
    s = s[0:no]

    # express shapes as coefficients
    K = beta.shape[2]
    VM = mean(v, 2)
    x = zeros((no, K))
    for ii in range(0, K):
        tmpv = v[:, :, ii]
        Utmp = U.T
        x[:, ii] = Utmp.dot((tmpv.flatten() - VM.flatten()))

    # Get PD plots:
    pd_results = []
    if pd_plots:
        VM = VM.flatten()
        for j in range(0, no):
            for i in range(1, 11):

                tmp = VM + 0.5 * (i - 5) * sqrt(s[j]) * U[:, j]
                m, n = q_mean.shape
                tmp_scale = 1
                v1 = tmp.reshape(m, n)
                q2n = cf.elastic_shooting(q_mean, v1, 1)

                p = cf.q_to_curve(q2n, tmp_scale)

                mv = 0.2

                pd_results.append([mv * i + p[0, :], p[1, :]])

    return x, pd_results

def formatData(dataFrame, varName="Name", xName="X", yName="Y"):
    # Pandas dataframe should have a column called "Name", "X","Y"

    names = pd.unique(dataFrame[varName])

    cont_length = len(dataFrame[dataFrame[varName] == names[0]][xName])
    total_cont = len(names)

    beta = np.zeros((2, cont_length, total_cont))

    for i, nm in enumerate(names):
        beta[0, :, i] = list(dataFrame[dataFrame[varName] == nm][xName])
        beta[1, :, i] = list(dataFrame[dataFrame[varName] == nm][yName])

    return beta

#### FDASRSF Functions

class fdacurve:
    """
    This class provides alignment methods for open and closed curves using the SRVF framework

    Usage: obj = fdacurve(beta, mode, N, scale)
    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param mode: Open ('O') or closed curve ('C') (default 'O')
    :param N: resample curve to N points
    :param scale: scale curve to length 1 (true/false)
    :param q:        (n,T,K) matrix defining n dimensional srvf on T samples with K srvfs
    :param betan:     aligned curves
    :param qn:        aligned srvfs
    :param basis:     calculated basis
    :param beta_mean: karcher mean curve
    :param q_mean:    karcher mean srvf
    :param gams:      warping functions
    :param v:         shooting vectors
    :param C:         karcher covariance
    :param s:         pca singular values
    :param U:         pca singular vectors
    :param coef:      pca coefficients
    :param qun:       cost function
    :param samples:   random samples
    :param gamr:      random warping functions
    :param cent:      center
    :param scale:     scale
    :param E:         energy

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  26-Aug-2020
    """

    def __init__(self, beta1, mode="O", scale=False):
        """
        fdacurve Construct an instance of this class
        :param beta: (n,T,K) matrix defining n dimensional curve on T samples with K curves
        :param mode:  Open ('O') or closed curve ('C') (default 'O')
        :param N: resample curve to N points
        :param scale: scale curve to length 1 (true/false)
        """
        self.mode = mode
        self.scale = scale

        K = beta1.shape[2]
        n = beta1.shape[0]
        N = beta1.shape[1]
        q = zeros((n, N, K))
        for ii in range(0, K):
            a = -cf.calculatecentroid(beta1[:, :, ii])
            beta1[:, :, ii] += tile(a, (N, 1)).T
            try:
                q[:, :, ii] = cf.curve_to_q(beta1[:, :, ii], self.scale, self.mode)
            except:
                q[:, :, ii] = cf.curve_to_q(beta1[:, :, ii], self.mode)[0]

        self.q = q
        self.beta = beta1

    def karcher_mean(self, parallel=True, cores=-1, method="DP"):
        """
        This calculates the mean of a set of curves
        :param parallel: run in parallel (default = F)
        :param cores: number of cores for parallel (default = -1 (all))
        :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"
        """
        n, T, N = self.beta.shape

        modes = ["O", "C"]
        mode = [i for i, x in enumerate(modes) if x == self.mode]
        if len(mode) == 0:
            mode = 0
        else:
            mode = mode[0]

        # Initialize mu as one of the shapes
        mu = self.q[:, :, 0]
        betamean = self.beta[:, :, 0]
        itr = 0

        gamma = zeros((T, N))
        maxit = 20

        sumd = zeros(maxit + 1)
        v = zeros((n, T, N))
        normvbar = zeros(maxit + 1)

        delta = 0.5
        tolv = 1e-4
        told = 5 * 1e-3

        print("Computing Karcher Mean of %d curves in SRVF space.." % N)
        while itr < maxit:
            print("updating step: %d" % (itr + 1))

            if iter == maxit:
                print("maximal number of iterations reached")

            mu = mu / sqrt(cf.innerprod_q2(mu, mu))
            if mode == 1:
                self.basis = cf.find_basis_normal(mu)
            else:
                self.basis = []

            sumv = zeros((n, T))
            sumd[0] = inf
            sumd[itr + 1] = 0
            out = Parallel(n_jobs=cores)(
                delayed(karcher_calc)(
                    self.beta[:, :, n],
                    self.q[:, :, n],
                    betamean,
                    mu,
                    self.basis,
                    mode,
                    method,
                )
                for n in range(N)
            )
            v = zeros((n, T, N))
            for i in range(0, N):
                v[:, :, i] = out[i][0]
                sumd[itr + 1] = sumd[itr + 1] + out[i][1] ** 2

            sumv = v.sum(axis=2)

            # Compute average direction of tangent vectors v_i
            vbar = sumv / float(N)

            normvbar[itr] = sqrt(cf.innerprod_q2(vbar, vbar))
            normv = normvbar[itr]

            if normv > tolv and fabs(sumd[itr + 1] - sumd[itr]) > told:
                # Update mu in direction of vbar
                mu = (
                    cos(delta * normvbar[itr]) * mu
                    + sin(delta * normvbar[itr]) * vbar / normvbar[itr]
                )

                if mode == 1:
                    mu = cf.project_curve(mu)

                x = cf.q_to_curve(mu)
                a = -1 * cf.calculatecentroid(x)
                betamean = x + tile(a, [T, 1]).T
            else:
                break
            itr += 1

        self.q_mean = mu
        self.beta_mean = betamean
        self.v = v
        self.qun = sumd[0 : (itr + 1)]
        self.E = normvbar[0 : (itr + 1)]

        return

    def srvf_align(self, parallel=True, cores=-1, method="DP"):
        """
        This aligns a set of curves to the mean and computes mean if not computed
        :param parallel: run in parallel (default = F)
        :param cores: number of cores for parallel (default = -1 (all))
        :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"
        """
        n, T, N = self.beta.shape

        modes = ["O", "C"]
        mode = [i for i, x in enumerate(modes) if x == self.mode]
        if len(mode) == 0:
            mode = 0
        else:
            mode = mode[0]

        # find mean
        if not hasattr(self, "beta_mean"):
            self.karcher_mean()

        self.qn = zeros((n, T, N))
        self.betan = zeros((n, T, N))
        self.gams = zeros((T, N))
        C = zeros((T, N))
        centroid2 = cf.calculatecentroid(self.beta_mean)
        self.beta_mean = self.beta_mean - tile(centroid2, [T, 1]).T
        q_mu = cf.curve_to_q(self.beta_mean)
        # align to mean

        out = Parallel(n_jobs=-1)(
            delayed(cf.find_rotation_and_seed_coord)(
                self.beta_mean, self.beta[:, :, n], mode, method
            )
            for n in range(N)
        )

        for ii in range(0, N):
            self.gams[:, ii] = out[ii][3]
            self.qn[:, :, ii] = out[ii][1]
            self.betan[:, :, ii] = out[ii][0]

        return


#### Geodesic Distances Functions

def init_path_geod(beta1, beta2, T=100, k=5):
    r"""
    Initializes a path in :math:`\cal{C}`. beta1, beta2 are already
    standardized curves. Creates a path from beta1 to beta2 in
    shape space, then projects to the closed shape manifold.

    :param beta1: numpy ndarray of shape (2,M) of M samples (first curve)
    :param beta2: numpy ndarray of shape (2,M) of M samples (end curve)
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy ndarray
    :return alpha: a path between two q-functions
    :return beta:  a path between two curves
    :return O: rotation matrix

    """
    alpha = np.zeros((2, T, k))
    beta = np.zeros((2, T, k))

    dist, pathq, O = gd.geod_sphere(beta1, beta2, k)

    for tau in range(0, k):
        alpha[:, :, tau] = project_curve(pathq[:, :, tau])
        x = cf.q_to_curve(alpha[:, :, tau])
        a = -1*cf.calculatecentroid(x)
        beta[:, :, tau] = x + np.tile(a, [T, 1]).T

    return(alpha, beta, O)

def update_path(alpha, beta, gradE, delta, T=100, k=5):
    """
    Update the path along the direction -gradE

    :param alpha: numpy ndarray of shape (2,M) of M samples
    :param beta: numpy ndarray of shape (2,M) of M samples
    :param gradE: numpy ndarray of shape (2,M) of M samples
    :param delta: gradient paramenter
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy scalar
    :return alpha: updated path of srvfs
    :return beta: updated path of curves

    """
    for tau in range(1, k-1):
        alpha_new = alpha[:, :, tau] - delta*gradE[:, :, tau]
        alpha[:, :, tau] = project_curve(alpha_new)
        x = cf.q_to_curve(alpha[:, :, tau])
        a = -1*cf.calculatecentroid(x)
        beta[:, :, tau] = x + np.tile(a, [T, 1]).T

    return(alpha, beta)

def project_curve(qvals):
    q= deepcopy(qvals)

    n,T = q.shape
    if n==2:
        dt = 0.35
    if n==3:
        dt = 0.2
    epsilon = 1e-6

    iter = 1
    res = np.ones(n)
    J = np.zeros((n,n))

    s = np.linspace(0,1,T)
    try:
      

        qnew = q.copy()
        qnew = qnew / np.sqrt(cf.innerprod_q2(qnew,qnew))

        qnorm = np.zeros(T)
        G = np.zeros(n)
        C = np.zeros(300)
        while (norm(res) > epsilon):
            if iter >= 300:
                break

            # Jacobian
            for i in range(0,n):
                for j in range(0,n):
                    J[i,j] = 3 * np.trapz(qnew[i,:]*qnew[j,:],s)

            J += np.eye(n)

            for i in range(0,T):
                qnorm[i] = norm(qnew[:,i])

            # Compute the residue
            for i in range(0,n):
                G[i] = np.trapz(qnew[i,:]*qnorm,s)

            res = -G

            if (norm(res) < epsilon):
                break

            x = solve(J,res)
            C[iter] = norm(res)

            delG = cf.Basis_Normal_A(qnew)
            temp = np.zeros((n,T))
            for i in range(0,n):
                temp += x[i]*delG[i]*dt

            qnew += temp
            iter += 1

        qnew = qnew/np.sqrt(cf.innerprod_q2(qnew,qnew))
        
    except:
        if q[0][0] == 0:
            q[0][0] = 1e-9
            
        qnew = q.copy()
        qnew = qnew / np.sqrt(cf.innerprod_q2(qnew,qnew))

        qnorm = np.zeros(T)
        G = np.zeros(n)
        C = np.zeros(300)
        while (norm(res) > epsilon):
            if iter >= 300:
                break

            # Jacobian
            for i in range(0,n):
                for j in range(0,n):
                    J[i,j] = 3 * np.trapz(qnew[i,:]*qnew[j,:],s)

            J += np.eye(n)

            for i in range(0,T):
                qnorm[i] = norm(qnew[:,i])

            # Compute the residue
            for i in range(0,n):
                G[i] = np.trapz(qnew[i,:]*qnorm,s)

            res = -G

            if (norm(res) < epsilon):
                break

            x = solve(J,res)
            C[iter] = norm(res)

            delG = cf.Basis_Normal_A(qnew)
            temp = np.zeros((n,T))
            for i in range(0,n):
                temp += x[i]*delG[i]*dt

            qnew += temp
            iter += 1

        qnew = qnew/np.sqrt(cf.innerprod_q2(qnew,qnew))
    
    return qnew

def path_straightening(beta1, beta2, betamid, init="rand", T=100, k=5):
    """
    Perform path straightening to find geodesic between two shapes in either
    the space of closed curves or the space of affine standardized curves.
    This algorithm follows the steps outlined in section 4.6 of Salili-James et al. 2022

    :param beta1: numpy ndarray of shape (2,M) of M samples (first curve)
    :param beta2: numpy ndarray of shape (2,M) of M samples (end curve)
    :param betamid: numpy ndarray of shape (2,M) of M samples (mid curve
     Default = NULL, only needed for init "rand")
    :param init: initialize path geodesic or random (Default = "rand")
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy ndarray
    :return dist: geodesic distance
    :return path: geodesic path
    :return pathsqnc: geodesic path sequence
    :return E: energy

    """
    inits = ["rand", "geod"]
    init = [i for i, x in enumerate(inits) if x == init]
    if len(init) == 0:
        init = 0
    else:
        init = init[0]

    betanew1, qnew1, A1 = pre_proc_curve(beta1)
    betanew2, qnew2, A2 = pre_proc_curve(beta2)

    if init == 0:
        betanewmid, qnewmid, Amid = pre_proc_curve(beta2)

    if init == 0:
        alpha, beta, O = gd.init_path_rand(betanew1, betanewmid,
                                        betanew2, T, k)
    elif init == 1:
        alpha, beta, O = init_path_geod(betanew1, betanew2, T, k)

    # path straightening
    tol = 1e-2
    n = beta.shape[0]
    T = beta.shape[1]
    maxit = 20
    i = 0
    g = 1
    delta = 0.5
    E = np.zeros(maxit+1)
    gradEnorm = np.zeros(maxit+1)
    pathsqnc = np.zeros((n, T, k, maxit+1))

    pathsqnc[:, :, :, 0] = beta

    while i < maxit:
        # algorithm 8:
        # compute dalpha/dt along alpha using finite difference approx
        # First calculate basis for normal sapce at each point in alpha
        basis = gd.find_basis_normal_path(alpha, k)
        alphadot = gd.calc_alphadot(alpha, basis, T, k)
        E[i] = gd.calculate_energy(alphadot, T, k)

        # algorithm 9:
        # compute covariant integral of alphadot along alpha. This is
        # the gradient
        # of E in \cal{H}. Later we will project it to the space \cal{H}_{O}
        u1 = gd.cov_integral(alpha, alphadot, basis, T, k)

        # algorithm 10:
        # backward parallel transport of u(1)
        utilde = gd.back_parallel_transport(u1[:, :, -1], alpha, basis, T, k)

        # algorithm 11:
        # compute gradient vector field of E in \cal{H}_{O}
        gradE, normgradE = gd.calculate_gradE(u1, utilde, T, k)
        gradEnorm[i] = norm(normgradE)
        g = gradEnorm[i]

        # algorithm 12:
        # update the path along the direction -gradE
        alpha, beta = update_path(alpha, beta, gradE, delta, T, k)

        # path evolution
        pathsqnc[:, :, :, i+1] = beta

        if g < tol:
            break

        i += 1

    if i > 0:
        E = E[0:i]
        gradEnorm = gradEnorm[0:i]
        pathsqnc = pathsqnc[:, :, :, 0:(i+2)]
    else:
        E = E[0]
        gradEnorm = gradEnorm[0]
        pathsqnc = pathsqnc[:, :, :, 0:(i+2)]

    path = beta
    dist = gd.geod_dist_path_strt(beta, k)

    return(dist, path, pathsqnc, E)

def pre_proc_curve(beta):
    """
    This function prepcoessed a curve beta to set of closed curves

    :param beta: numpy ndarray of shape (2,M) of M samples
    :param T: number of samples (default = 100)

    :rtype: numpy ndarray
    :return betanew: projected beta
    :return qnew: projected srvf
    :return A: alignment matrix (not used currently)

    """
    T = np.shape(beta)[1]
    try:
        q = cf.curve_to_q(beta)[0]
        qnew = project_curve(q)
    except:
        q = cf.curve_to_q(beta)
        qnew = project_curve(q)
    x = cf.q_to_curve(qnew)
    a = -1 * cf.calculatecentroid(x)
    betanew = x + np.tile(a, [T, 1]).T
    A = np.eye(2)

    return (betanew, qnew, A)


def geodDistance(X1,Y1,X2,Y2,k=5):

    F1 = np.column_stack((X1,Y1)).T
    F2 = np.column_stack((X2,Y2)).T
    T = len(X1)
    try:
        d,path,pathsq,E = path_straightening(F1,F2,'NULL',init="geod",T=T,k=int(k))
    except:
        d,path,pathsq,E = path_straightening(F2,F1,'NULL',init="geod",T=T,k=int(k))

    return d,path,pathsq,E

#### KNN Functions

def modeValue(values, valuesOrder=[]):
    try:
        # Compute mode of list using the mode function from the statistics library.
        # This function can only output a mode if there is a unique mode.
        md = mode(values)
    except:
        # Find multiple mode values.
        counts = [values.count(i) for i in np.unique(values)]
        modes = np.unique(values)[np.where(np.array(counts) == max(counts))[0]]
        # Now, we either choose the top mode based on some rank, or simply return all mode values:
        if len(valuesOrder) > 0:
            # Choose the mode value that has the lowest rank in the order (e.g. shortest distances)
            k = np.argmin([np.where(np.array(valuesOrder) == p)[0][0] for p in modes])
            md = modes[k]
        else:
            md = modes
    return md


def KNN_distMat(
    distanceMatrix,
    metaData,
    trainingNames,
    testingNames,
    classType=str,
    neighbours=[5],
    f1Average="weighted",
    inMeta=True,
    nameVariable="Name",
    classVariable="Class"
):
    # This function runs a KNN classifier on a training and testing set.
    # K is equal to the neighbours variable.
    # If more than one value is passed in the neighbours variable, then
    # we find the KNN F1 scores using all the neighbours in the list,
    # and output the KNN results that obtained the highest F1 score.
    f1Score = 0
    topNeighbour = neighbours[0]

    distanceMatrix = distanceMatrix.set_index("Filename")

    for neigh in neighbours:

        # This will be updated during the Knn algorithm to contain the list of all the pots and their predicted sample.
        # To start with, it only contains the details from the training sample.
        classDetails = {}
        for name in trainingNames:
            classDetails.update(
                {name: {"Class": list(metaData[metaData[nameVariable] == name][classVariable])[0]}}
            )

        for testName in testingNames:
            dists = []
            # Find the distances between current test value and all the training set.
            for trainName in trainingNames:
                d = distanceMatrix[testName][trainName]
                dists.append(d)
            # Sort the distances from smallest to largest.
            topRank = np.argsort(dists)[:neigh]
            topClasses = []
            for ind in topRank:
                topClasses.append(classDetails[trainingNames[ind]]["Class"])
            # If there are multiple choices for the mode, we pick the one with the smallest distance.
            shapeclass = modeValue(topClasses, topClasses)
            classDetails.update({testName: {"Class": shapeclass}})

        if inMeta == True:
            # If the test set is included in the meta data, then we can compute F1 scores
            # by using the ground truth values in the meta data.
            actual = []
            predicted = []
            for testName in testingNames:
                pred = classType(classDetails[testName]["Class"])
                act = classType(
                    list(metaData[metaData[nameVariable] == testName][classVariable])[0]
                )
                actual.append(act)
                predicted.append(pred)

            f1 = f1_score(actual, predicted, average=f1Average)

            # If only one neighbour was selected in the neighbour variable, then we output
            # the predicted and actual values from that k. Else, we output the results
            # that obtained the highest F1 score.
            if len(neighbours) > 1:
                if f1 >= f1Score:
                    f1Score = deepcopy(f1)
                    topNeighbour = deepcopy(neigh)
                    predictedClass = deepcopy(predicted)
                    actualClass = deepcopy(actual)
                    results = pd.DataFrame(
                        [testingNames, predictedClass, actualClass]
                    ).T
                    results = results.rename(
                        columns={0: "Name", 1: "predictedClass", 2: "actualClass"}
                    )
            else:
                f1Score = deepcopy(f1)
                predictedClass = deepcopy(predicted)
                actualClass = deepcopy(actual)
                results = pd.DataFrame([testingNames, predictedClass, actualClass]).T
                results = results.rename(
                    columns={0: "Name", 1: "predictedClass", 2: "actualClass"}
                )
        else:
            # If the meta data doesn't include the test set, then we output the results
            # (predicted classes) from the last neighbour and the F1 score is set to "N/A".
            predictedClass = []
            for testName in testingNames:
                pred = classType(classDetails[testName]["Class"])
                predictedClass.append(pred)
            f1Score = "N/A"
            results = pd.DataFrame([testingNames, predictedClass]).T
            results = results.rename(columns={0: "Name", 1: "predictedClass"})

    return results, f1Score, topNeighbour


def KNN_Bootstrapping(
    table,
    all_dists,
    classes,
    kNeighbours=np.int_(np.linspace(3, 12, 10)),
    nBootstrapping=50,
    classType=str,
    trainingSetSize=12,
    trainingProportion=0,
    nameVariable="Name",
    classVariable="Class"
):

    all_scores = np.zeros((1, nBootstrapping))
    top_neighs = np.zeros((1, nBootstrapping))
    all_classification = []

    top_score = 0
    top_results = []
    top_trainingTesting = []

    all_training = []

    for nb in range(0, nBootstrapping):

        # 1) Create training and test sets:
        train_names = []
        test_names = []

        for n, sp in enumerate(classes):
            inds = list(table[table[classVariable] == sp]["Index"])
            if trainingProportion != 0:
                trainingSetSize = int(len(inds) * trainingProportion)
            randinds = random.sample(inds, trainingSetSize)
            testnames = list(table[nameVariable][list(np.setdiff1d(inds, randinds))])
            trainnames = list(table[nameVariable][randinds])
            train_names.extend(trainnames)
            test_names.extend(testnames)
        all_training.append(train_names)

        results, f1Score, topNeighbour = KNN_distMat(
            all_dists,
            table,
            train_names,
            test_names,
            classType=classType,
            neighbours=kNeighbours,
            f1Average="weighted",
            inMeta=True,
            nameVariable=nameVariable,
            classVariable=classVariable
        )

        all_scores[0, nb] = f1Score
        top_neighs[0, nb] = topNeighbour
        all_classification.append(results)

        if f1Score >= top_score:
            top_score = deepcopy(f1Score)
            top_results = deepcopy(results)
            top_trainingTesting = {'train':train_names,'test':test_names}

    all_results = {}
    all_results["topF1Score"] = top_score
    all_results["topClassification"] = top_results
    all_results["topTrainingTesting"] = top_trainingTesting
    all_results["allF1Scores"] = all_scores
    all_results["allTopNeighbours"] = top_neighs
    all_results["allTrainingSamples"] = all_training
    all_results["allClassificationResults"] = all_classification

    return all_results


#### Other Functions

def coords_corresp(x,y):
    k = np.argmin(y) # returns the minimum value of y
    new_y = []
    new_x = []
    new_y.extend(y[k:])
    new_y.extend(y[:k+1])
    new_x.extend(x[k:])
    new_x.extend(x[:k+1])

    return new_x, new_y

def rescale(x, y, scaledHeightWidth=3, scaleDirection="both"):
    # This function rescales the contour so that it's between certain values (of a certain width/height)
    # and centred at (0,0). The 'scaleDirection' parameter is set to "both" as defult. This means that
    # it scales the max(height,width) to the desired width/height. The smaller side (heigh/width) will
    # then be scaled with the same scale factor, and hence lie within the desired range (and not greater).
    # If it's desired that the height will be of a certain length, or the width, then set the
    # scaleDirection paramter to "height" or "width" respectively.
    # Input: x, y coordinates, desired max height/width, and desired direction (height/width/max)
    # Output: rescaled x and y coordinates.

    if type(x) != np.ndarray:
        x = np.array(x)

    if type(y) != np.ndarray:
        y = np.array(y)

    if scaleDirection == "height":
        p = max(y) - min(y)
    elif scaleDirection == "width":
        p = max(x) - min(x)
    else:
        p = max(max(y) - min(y), max(x) - min(x))

    q = p / scaledHeightWidth

    x_ = x / q
    y_ = y / q
    mp = min(x_) + (max(x_) - min(x_)) / 2
    x_ = x_ - mp
    mp = min(y_) + (max(y_) - min(y_)) / 2
    y_ = y_ - mp

    return x_, y_