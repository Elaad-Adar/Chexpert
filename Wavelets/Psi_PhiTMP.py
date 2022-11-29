import numpy as np


def Psi_PhiTMP(N, Par, DeTr):
    # fpsi:  DFTs of discrete-time-spline wavelet packets in the space Pi[N], N=2**j
    # psi: discrete-spline wavelet packets in the space Pi[N], N=2**j
    # fphi:  DFTs of discrete-time-spline complementary wavelet packets in the space Pi[N], N=2**j
    # phi: discrete-time-spline complementary  wavelet packets in the space Pi[N], N=2**j
    # z: positive complex qWP; fz: its DFT
    # zm: negative complex qWP; fzm: its DFT
    # DeTr: depth of transform
    # Par: spline orderPsi_PhiTMP.py

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
    # KK = (0, 1)
    KK = [0, 1]
    [alpha, beta] = alpha_betaTMP(M, Par)  # computation coeffs of modulation matrices for discrete-spline qWPs

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
    fpsi_lev[0, :] = fpsi_lev[0, :] / np.linalg.norm(psi_lev[0, :])  # todo: should it be /np.linalg.norm(fpsi_lev[0, :])?
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
    fpsi_lev[1, :] = fpsi_lev[1, :] / np.linalg.norm(psi_lev[1, :])  # todo:  should it be /np.linalg.norm(fpsi_lev[1, :])?
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

        alpha, beta = alpha_betaTMP(M, Par)
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


def alpha_betaTMP(N,p):
    # alpha, beta: coeffs. of modulation matrices for  f1'*f2-g1'*g2 wavelet transforms
    # N: length of row signals
    # p: spline order

    n = np.arange(N)
    nn = np.arange(N//2)
    ep = np.exp(2 * np.pi * 1j * n / N)

    em = np.exp(-2 * np.pi * 1j * n / N)
    [_, fbsp, fbspm, UPS] = per_dbspTZsk(N // 2, p)

    UPS = np.hstack((UPS, UPS))

    beta = fbsp / np.sqrt(1 * UPS)
    alpha = fbspm / np.sqrt(1 * UPS) * ep
    return alpha, beta


def per_dbspTZsk(period,order):
    # Computation of the discrete-time B-spline `bsp` of any order, its DFT `fbsp`,  DFT of its odd samples 'fbspm' and  the characteristic sequence `UPS'

    N = period

    Par = order
    n = np.arange(2 * N)
    nu = np.arange(1, N)
    w = np.exp(-1 * np.pi * 1j * n / N)
    u, v = juviTSsk(Par, int(N))
    #v2 = np.array([v, v])  # should stack v, v on axis 1
    #u2 = np.array([u, u])  # should stack u, u on axis 1
    v2 = np.hstack((v, v))
    u2 = np.hstack((u, u))
    wv = np.real(v2 * w)
    fbsp = (wv + u2) / 2
    fbspm = (-wv + u2) / 2
    bsp = np.real(np.fft.ifft(fbsp))

    # V = v. / u;
    UPS = (u ** 2 + abs(v) ** 2) / 4

    return bsp, fbsp, fbspm, UPS


def juviTSsk(Par,N):
    # u: characteristic sequence of the space of 'N'-periodic polynomial splines of order 'Par'
    # (DFT of the span-2 B-spline sampled at even grid points)
    # v: DFT of the span-2 B-spline sampled at odd grid points
    # import json
    #
    # with open('/Users/elaadadar/Desktop/Automation/Thesis/ChexPert/waveletNN/python_Gil/config.json', 'r') as f:
    #     config = json.load(f)
    #
    #
    # if config['colab']:
    BSU = np.loadtxt("Wavelets/BSU.txt")  # samples of the B-splines

    if Par == 0:
        Par = 1
    V = BSU[Par - 1, 15 + np.arange(Par)]
    U = BSU[Par - 1, :Par]
    dv = np.where(np.diff(V, axis=0) == 0)[0][0] + 1
    vv = np.zeros(max(Par, N), dtype='complex')  # todo: check max(Par, N)
    # vv = np.zeros((1, N), dtype='complex')
    # vv.reshape(-1)[:Par]
    vv[:Par] = V
    vv = np.roll(vv, -dv)
    u = np.abs(np.fft.fft(U, n=N))
    v = np.fft.fft(vv, n=N)
    return u, v




