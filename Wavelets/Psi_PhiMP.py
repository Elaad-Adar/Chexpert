import numpy as np


def Psi_PhiMP(N, Par, DeTr):
    # fpsi:  DFTs of discrete-spline wavelet packets in the space Pi[N], N=2**j
    # psi: discrete-spline wavelet packets in the space Pi[N], N=2**j
    # fphi:  DFTs of discrete-spline complementary wavelet packets in the space Pi[N], N=2**j
    # phi: discrete-spline complementary  wavelet packets in the space Pi[N], N=2**j
    # z: positive complex qWP; fz: its DFT
    # zm: negative complex qWP; fzm: its DFT
    # DeTr: depth of transform
    # Par: spline order

    n = 2 * (2 ** DeTr - 1)
    fpsi = np.zeros((n, N), dtype='complex')
    psi = fpsi.copy()
    z = psi.copy()
    fz = psi.copy()
    zm = psi.copy()
    fzm = psi.copy()

    phi = psi.copy()
    fphi = psi.copy()
    M = N
    K = 2
    KK = (0, 1)
    [alpha, beta] = alpha_betaMP(M, Par)  # computation coeffs of modulation matrices for discrete-spline qWPs

    fpsi_lev = np.zeros((2, N), dtype='complex')
    psi_lev = fpsi_lev.copy()
    fphi_lev = fpsi_lev.copy()
    phi_lev = fpsi_lev.copy()
    fz_lev = fpsi_lev.copy()
    z_lev = fpsi_lev.copy()
    fzm_lev = fpsi_lev.copy()
    zm_lev = fpsi_lev.copy()

    fpsi_lev[0, :] = beta
    psi_lev[0, :] = np.real(np.fft.ifft(fpsi_lev[0, :]))
    psi_lev[0, :] = psi_lev[0, :] / np.linalg.norm(psi_lev[0, :])
    fpsi_lev[0, :] = fpsi_lev[0, :] / np.linalg.norm(psi_lev[0, :])
    fz_lev[0, :] = np.zeros((1, M))
    fz_lev[0, 0] = (1 + 1j) * fpsi_lev[0, 0]
    fz_lev[0, M // 2] = (1 + 1j) * fpsi_lev[0, M // 2]
    fz_lev[0, 1: M // 2] = 2 * fpsi_lev[0, 1: M // 2]
    z_lev[0, :] = np.fft.ifft(fz_lev[0, :])
    phi_lev[0, :] = np.imag(z_lev[0, :])
    phi_lev[0, :] = phi_lev[0, :] / np.linalg.norm(phi_lev[0, :])
    fphi_lev[0, :] = np.fft.fft(phi_lev[0, :])
    fzm_lev[0, :] = fpsi_lev[0, :] - 1j * fphi_lev[0, :]
    zm_lev[0, :] = psi_lev[0, :] - 1j * phi_lev[0, :]

    fpsi_lev[1, :] = alpha

    psi_lev[1, :] = np.real(np.fft.ifft(fpsi_lev[1, :]))
    psi_lev[1, :] = psi_lev[1, :] / np.linalg.norm(psi_lev[1, :])
    fpsi_lev[1, :] = fpsi_lev[1, :] / np.linalg.norm(psi_lev[1, :])
    # fz_lev[1, :] = np.zeros(2, M)
    fz_lev[1, 0] = (1 + 1j) * fpsi_lev[1, 0]
    fz_lev[1, M // 2] = (1 + 1j) * fpsi_lev[1, M // 2]
    fz_lev[1, 1: M // 2] = 2 * fpsi_lev[1, 1: M // 2]
    z_lev[1, :] = np.fft.ifft(fz_lev[1, :])
    phi_lev[1, :] = np.imag(z_lev[1, :])
    phi_lev[1, :] = phi_lev[1, :] / np.linalg.norm(phi_lev[1, :])
    fphi_lev[1, :] = np.fft.fft(phi_lev[1, :])
    fzm_lev[1, :] = fpsi_lev[1, :] - 1j * fphi_lev[1, :]
    zm_lev[1, :] = psi_lev[1, :] - 1j * phi_lev[1, :]

    fpsi[:2, :] = fpsi_lev
    psi[:2, :] = psi_lev
    fphi[:2, :] = fphi_lev
    phi[:2, :] = phi_lev
    fz[:2, :] = fz_lev
    z[:2, :] = z_lev
    fzm[:2, :] = fzm_lev
    zm[:2, :] = zm_lev

    for kk in range(1, DeTr):
        M = M // 2
        mn = N // M
        K = 2 ** (kk + 1)
        fpsi_lev = np.zeros((K, N), dtype='complex')

        psi_lev = fpsi_lev.copy()
        fphi_lev = fpsi_lev.copy()
        phi_lev = fpsi_lev.copy()
        fz_lev = fpsi_lev.copy()
        z_lev = fpsi_lev.copy()
        fzm_lev = fpsi_lev.copy()
        zm_lev = fpsi_lev.copy()

        alpha, beta = alpha_betaMP(M, Par)
        alpha = np.tile(alpha, (1, mn))
        beta = np.tile(beta, (1, mn))

        for nn in range(K // 2):
            yy = fz[KK, :]
            y = yy[nn, :]
            # y = fz(kk - 1}{nn}; ???
            y1 = y * alpha
            y0 = y * beta
            s = nn % 2
            fz_lev[2 * nn, :] = y1 * s + y0 * (1 - s)
            fz_lev[2 * nn + 1, :] = y0 * s + y1 * (1 - s)
            z_lev[2 * nn, :] = np.fft.ifft(fz_lev[2 * nn, :])
            z_lev[2 * nn + 1, :] = np.fft.ifft(fz_lev[2 * nn + 1, :])
            psi_lev[2 * nn, :] = np.real(z_lev[2 * nn, :])
            psi_lev[2 * nn + 1, :] = np.real(z_lev[2 * nn + 1, :])
            fpsi_lev[2 * nn, :] = np.fft.fft(psi_lev[2 * nn, :])
            fpsi_lev[2 * nn + 1, :] = np.fft.fft(psi_lev[2 * nn + 1, :])

            phi_lev[2 * nn, :] = np.imag(z_lev[2 * nn, :])
            phi_lev[2 * nn + 1, :] = np.imag(z_lev[2 * nn + 1, :])
            fphi_lev[2 * nn, :] = np.fft.fft(phi_lev[2 * nn, :])
            fphi_lev[2 * nn + 1, :] = np.fft.fft(phi_lev[2 * nn + 1, :])

            fzm_lev[2 * nn, :] = fpsi_lev[2 * nn, :] - 1j * fphi_lev[2 * nn, :]
            fzm_lev[2 * nn + 1, :] = fpsi_lev[2 * nn + 1, :] - 1j * fphi_lev[2 * nn + 1, :]
            zm_lev[2 * nn, :] = psi_lev[2 * nn, :] - 1j * phi_lev[2 * nn, :]
            zm_lev[2 * nn + 1, :] = psi_lev[2 * nn + 1, :] - 1j * phi_lev[2 * nn + 1, :]

        KK = KK[-1] + np.arange(K) + 1
        fpsi[KK, :] = fpsi_lev
        psi[KK, :] = psi_lev
        fphi[KK, :] = fphi_lev
        phi[KK, :] = phi_lev
        fz[KK, :] = fz_lev
        z[KK, :] = z_lev
        fzm[KK, :] = fzm_lev
        zm[KK, :] = zm_lev

    return psi, fpsi, phi, fphi, z, fz, zm, fzm


def alpha_betaMP(N, Par):
    # alpha, beta: parameters for discrete-spline wavelet packet transform and wavelet packet design
    n = np.arange(N)
    nn = np.arange(N/2)
    ep = np.exp(2 * np.pi * 1j * n / N)
    em = np.exp(-2 * np.pi * 1j * n / N)
    co = (np.cos(np.pi * n / N)) ** Par
    si = (np.sin(np.pi * n / N)) ** Par
    U = np.sqrt((co ** 2 + si ** 2) / 2)
    beta = co / U
    alpha = si / U * ep
    return alpha, beta




