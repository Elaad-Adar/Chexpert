import numpy as np
from Wavelets.havshB import havshB
from Wavelets.Psi_PhiMP import Psi_PhiMP
from Wavelets.Psi_PhiTMP import Psi_PhiTMP


def matmMP(N, Par, DeTr, pm, dc):
    # building modulation matrices MatrMz for direct qWP transforms
    # dc=1: discrete-spline wavelet transform; dc=2: discrete-time-spline wavelet transform
    # pm=1: regular transform, pm==2: complementary transform
    # pm=3: positive qWP transform, pm=4: negative qWP transform
    # Par: spline order
    # DeTr: depth of transform

    MatrMz = np.zeros((4 * DeTr, N // 2), dtype='complex')

    tM, _ = MmatrMP(N, Par, pm, dc)

    MatrMz[:4, :] = tM
    for kk in range(1, DeTr):
        N = N // 2
        tM, _ = MmatrMP(N, Par, 1, dc)
        rows_mask = kk * 4 + np.arange(4)
        MatrMz[rows_mask, :N // 2] = tM

    return MatrMz


def MmatrMP(N, Par, pm, dc):
    # tM: modulation matrix for direct discrete-spline (for dc=1) or discrete-time-spline (for dc=2) wavelet packet transform
    # M: modulation matrix for inverse transform
    # pm=1: regular transform, pm=2: complementary transform
    # pm=3: positive qWP transform, pm=4: negative qWP transform
    # N: length of row signals
    # Par: spline order
    assert dc in (1, 2)
    if dc == 1:
        [_, fpsi, _, fphi, _, fz, _, fzm] = Psi_PhiMP(N, Par, 1)  # computation of DFT of WPs
    else:  # dc == 2
        [_, fpsi, _, fphi, _, fz, _, fzm] = Psi_PhiTMP(N, Par, 1)


    p11, p12 = havshB(fphi[0, :])
    p21, p22 = havshB(fphi[1, :])
    P = np.vstack((p11, p21, p12, p22))
    s11, s12 = havshB(fpsi[0, :])
    s21, s22 = havshB(fpsi[1, :])
    S = np.vstack((s11, s21, s12, s22))
    q11, q12 = havshB(fz[0, :])
    q21, q22 = havshB(fz[1, :])
    Q = np.vstack((q11, q21, q12, q22))

    t11, t12 = havshB(fzm[0, :])
    t21, t22 = havshB(fzm[1, :])
    T = np.vstack((t11, t21, t12, t22))
    assert pm in (1, 2, 3, 4)
    if pm == 1:
        tM = np.conj(S)
        M = S
    elif pm == 2:
        tM = np.conj(P)
        M = P
    elif pm == 3:
        tM = np.conj(Q)
        M = Q
    else:  # pm == 4
        tM = np.conj(T)
        M = T

    return tM, M
