import numpy as np
from matplotlib import pyplot as plt

from havshB import havshB



def WP_an_2dMP(x, DeTr, MatrM_pv,MatrM_ph,MatrM_m):
    # 2D discrete-(time-)spline quasi-analytic wavelet packet decomposition of the square array `x`
    # MatrMzp and MatrMzm: modulation matrices for "positive" and  "negative" transforms
    # ptran and mtran are transform coeff. from levels from 1 to DeTr  for "positive" and "negative" transforms
    # DeTr: depth of decomposition

    # TODO find out why ptran is with both positive mat and mtran is one positive and one negative?
    ptran = dsp_wq_an_2d_sk(x, DeTr, MatrM_ph, MatrM_pv)
    mtran = dsp_wq_an_2d_sk(x, DeTr, MatrM_ph, MatrM_m)


    return ptran, mtran


def dsp_wq_an_2d_sk(x, DeTr, Mat1, Mat2):
    # 2D discrete-spline wavelet packet analysis of the square array `x`
    # DeTr: depth of decomposition
    Nv, Nh=x.shape
    wqtran = np.zeros((Nv,Nh, DeTr), dtype='complex')
    k_lev = np.zeros((Nv,Nh), dtype='complex')
    z0, z1 = dsw_down_sk(x, Mat1[:4, :])
    y0 = z0.T #(128, 256)
    y1 = z1.T #(128, 256)
    z0, z1 = dsw_down_sk(y0, Mat2[:4, :])
    y00 = z0.T
    y10 = z1.T
    z0, z1 = dsw_down_sk(y1, Mat2[:4, :])
    y01 = z0.T
    y11 = z1.T

    k_lev = np.vstack((np.hstack((y00, y01)),
                       np.hstack((y10, y11))))
    wqtran[:, :, 0] = k_lev

    for kk in range(1, DeTr):
        K = 2 ** (kk + 1)
        MAh = Mat1[(kk*4):(kk*4+4), : Nv // K]
        MAv = Mat1[(kk*4):(kk*4+4), : Nh // K]

        k_lev = np.zeros((Nv, Nh), dtype='complex')
        for nn in range(K // 2):
            for mm in range(K // 2):
                mask_rows_start = nn * Nv * 2 // K
                amountv = 2 * Nv // K
                mask_rows_end = mask_rows_start+amountv
                amounth = 2 * Nh // K

                mask_cols_start = mm * Nh * 2 // K
                mask_cols_end = mask_cols_start+amounth
                xx = wqtran[mask_rows_start: mask_rows_end, mask_cols_start:mask_cols_end, kk - 1]
                yy = xx

                s = (mm+1) % 2
                z0, z1 = dsw_down_sk(xx, MAh)
                t = (nn+1) % 2
                y0 = z1 * (1 - s) + z0 * s
                y1 = z0 * (1 - s) + z1 * s
                y0 = y0.T
                y1 = y1.T
                z0, z1 = dsw_down_sk(y0, MAv)
                y00 = (z1 * (1 - t) + z0 * t).T
                y10 = (z0 * (1 - t) + z1 * t).T
                z0, z1 = dsw_down_sk(y1, MAv)
                y01 = (z1 * (1 - t) + z0 * t).T
                y11 = (z0 * (1 - t) + z1 * t).T
                yy = np.vstack((np.hstack((y00, y01)),
                                np.hstack((y10, y11))))

                mask_rows_start = nn * Nv * 2 // K
                mask_rows_end = mask_rows_start+2 * Nv // K
                mask_cols_start = mm * Nh * 2 // K
                mask_cols_end = mask_cols_start + 2 * Nh // K
                k_lev[mask_rows_start:mask_rows_end, mask_cols_start:mask_cols_end] = yy

        wqtran[:, :, kk] = k_lev

    return wqtran

def dsw_down_sk(x, Mat):
    # one step of discrete-spline wavelet transform of a row signal or the array `x` of row signals
    # 2r: spline order
    # pm==1: regular transform, pm==2: complementary transform
    # pm==3: positive q_analytic transform, pm==4: negative q_analytic transform
    vx, hx = x.shape
    N = hx
    tM11 = Mat[0, :]
    tM21 = Mat[1, :]
    tM12 = Mat[2, :]
    tM22 = Mat[3, :]
    # tM, _ = MmatrZ(N, vx, r, pm)
    xx = np.fft.fft(x, axis=1)
    xx0, xx1 = havshB(xx)
    yy0 = xx0 * tM11 + xx1 * tM12
    yy1 = xx0 * tM21 + xx1 * tM22
    y0 = (np.fft.ifft(yy0, axis=1)) / 2
    y1 = (np.fft.ifft(yy1, axis=1)) / 2

    return y0, y1
