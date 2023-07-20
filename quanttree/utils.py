import numpy as np
import os


def pkg_folder():
    x, _ = os.path.split(__file__)
    return x


def random_gaussian(dim):
    mu0 = np.zeros(dim)
    sigma0 = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim)
    Q, _ = np.linalg.qr(sigma0)
    D = np.diag(2 * np.abs(np.random.normal(0, 1, dim)) + 0.01)
    sigma0 = np.dot(np.dot(Q, D), np.transpose(Q))
    return mu0, sigma0


def compute_roto_translation(gauss0, target_sKL=1):
    mu0 = gauss0[0]
    sigma0 = gauss0[1]
    dim = len(mu0)
    shift = np.zeros(dim)
    if dim == 1:
        rot = 1
        angles = 0
        num_angles = 1
        P = 1
    else:
        num_angles = int(np.floor(dim / 2))
        A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim)
        P, _ = np.linalg.qr(A)

        angles = -np.pi / 2 * np.random.rand(num_angles) + np.pi / 2
        Q = generate_canonical_rotation(angles, dim)
        rot = np.dot(np.dot(P, Q), np.transpose(P))

    gauss1 = rotate_and_shift_gaussian(gauss0, rot, shift)
    sKL = symmetric_kullback_leibler(gauss0, gauss1)

    versor = np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
    versor = versor / np.sqrt(np.sum(versor ** 2))

    to_decrease = sKL > target_sKL

    while to_decrease:
        angles = angles / 2
        Q = generate_canonical_rotation(angles, dim)
        rot = np.dot(np.dot(P, Q), np.transpose(P))

        gauss1 = rotate_and_shift_gaussian(gauss0, rot, shift)
        sKL = symmetric_kullback_leibler(gauss0, gauss1)
        to_decrease = sKL > target_sKL

    sigma1 = gauss1[1]
    I = np.identity(dim)

    a = np.dot(versor, np.linalg.solve(sigma1, versor)) + np.dot(versor, np.linalg.solve(sigma0, versor))
    b = np.dot(np.dot(versor, np.linalg.solve(sigma1, rot - I)), mu0) + np.dot(
        np.dot(versor, np.linalg.solve(sigma0, rot - I)), mu0)
    c = 2 * sKL - 2 * target_sKL
    rho = (-b + np.sqrt(b ** 2 - a * c)) / a
    shift = rho * versor

    return rot, shift


def rotate_and_shift_gaussian(gauss, rot, shift):
    mu0 = gauss[0]
    sigma0 = gauss[1]
    mu1 = np.dot(mu0, np.transpose(rot)) + shift
    sigma1 = np.dot(np.dot(rot, sigma0), np.transpose(rot))
    sigma1 = 0.5 * (sigma1 + np.transpose(sigma1))
    return mu1, sigma1


def rotate_gaussian(gauss, rot):
    mu0 = gauss[0]
    sigma0 = gauss[1]
    mu1 = np.dot(mu0, np.transpose(rot))
    sigma1 = np.dot(np.dot(rot, sigma0), np.transpose(rot))
    sigma1 = 0.5 * (sigma1 + np.transpose(sigma1))
    return mu1, sigma1


def generate_canonical_rotation(angles, dim):
    Q = np.identity(dim)
    if dim == 1:
        return Q

    for i_angles in range(len(angles)):
        theta = angles[i_angles]
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        Q[2 * i_angles: 2 * (i_angles + 1), 2 * i_angles: 2 * (i_angles + 1)] = R

    return Q


def generate_random_rotation(dim):
    if dim == 1:
        Q = 1
    else:
        num_angles = int(np.floor(dim / 2))

        angles = np.zeros(num_angles)
        angles[0] = np.random.uniform(0, 2 * np.pi)
        for i in range(num_angles - 1):
            angles[i + 1] = np.random.uniform(-np.pi, np.pi)

        Q = generate_canonical_rotation(angles, dim)

        A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim)
        P, _ = np.linalg.qr(A)
        Q = np.dot(P, np.dot(Q, np.transpose(P)))

    return Q


def kullback_leibler(gauss0, gauss1):
    mu0 = gauss0[0]
    sigma0 = gauss0[1]
    mu1 = gauss1[0]
    sigma1 = gauss1[1]
    dim = len(mu0)
    return 0.5 * (np.trace(np.linalg.solve(sigma1, sigma0)) + np.dot(mu1 - mu0,
                                                                     np.linalg.solve(sigma1, mu1 - mu0)) - dim + np.log(
        np.linalg.det(sigma1) / np.linalg.det(sigma0)))


def symmetric_kullback_leibler(gauss0, gauss1):
    return kullback_leibler(gauss0, gauss1) + kullback_leibler(gauss1, gauss0)


def compute_rotation(gauss0, target_sKL=1, nrun=100, maxiter=1000):
    mu0 = gauss0[0]
    sigma0 = gauss0[1]
    dim = len(mu0)
    shift = np.zeros(dim)
    sKL = 0
    run = 0
    for run in range(nrun):
        if dim == 1:
            rot = 1
            angles = 0
            num_angles = 1
            P = 1
        else:
            num_angles = int(np.floor(dim / 2))
            A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim)
            P, _ = np.linalg.qr(A)

            angles = -np.pi / 2 * np.random.rand(num_angles) + np.pi / 2
            Q = generate_canonical_rotation(angles, dim)
            rot = np.dot(np.dot(P, Q), np.transpose(P))

            gauss1 = rotate_and_shift_gaussian(gauss0, rot, shift)
            sKL = symmetric_kullback_leibler(gauss0, gauss1)
            if sKL >= target_sKL:
                break
    else:
        return None

    # proseguo con la bisezione
    theta_l = np.zeros(num_angles)
    theta_r = angles
    sKL_r = sKL
    sKL_l = 0

    for iter in range(maxiter):
        theta = (theta_l + theta_r) / 2

        Q = generate_canonical_rotation(theta, dim)
        rot = np.dot(np.dot(P, Q), np.transpose(P))
        gauss1 = rotate_and_shift_gaussian(gauss0, rot, shift)
        sKL = symmetric_kullback_leibler(gauss0, gauss1)

        if np.abs(sKL - target_sKL) < 1e-6:
            return rot
        if np.sign(sKL - target_sKL) == np.sign(sKL_l - target_sKL):
            sKL_l = sKL
            theta_l = theta
        else:
            sKL_r = sKL
            theta_r = theta

    return rot


def generate_gaussian_modalities(nmodes, dim, d0=1, check_max=False, dmax=5):
    counter = 0
    modes = []
    for i in range(nmodes):
        if len(modes) == 0:
            modes.append(random_gaussian(dim))
        else:
            new_mode = None
            accepted = False
            while not accepted:
                if counter == 25:
                    counter = 0
                    dmax = dmax * 2

                accepted = True
                rnd_mode = modes[np.random.choice(i)]
                rot, shift = compute_roto_translation(rnd_mode, d0)
                mu, sigma = rotate_and_shift_gaussian(rnd_mode, rot, shift)
                _, sigma = random_gaussian(dim)
                new_mode = (mu, sigma)

                for j in range(i):
                    dij = symmetric_kullback_leibler(new_mode, modes[j])
                    if dij < d0:
                        accepted = False
                        break
                    if check_max and dij > dmax:
                        accepted = False
                        counter = counter + 1
                        break
            modes.append(new_mode)

    return modes


def generate_multimodal_batches(modes, nbatches, nu):
    nmodes = len(modes)
    d = len(modes[0][0])

    data = np.zeros((nmodes, nbatches, nu, d)) * np.nan
    for m, mode in enumerate(modes):
        # data[m] = np.random.multivariate_normal(mode[0], mode[1], nbatches * nu).reshape((nbatches, nu, d))
        data[m] = np.random.multivariate_normal(mode[0], mode[1], (nbatches, nu))

    if np.sum(np.isnan(data)) > 0:
        raise Exception("NaN found!")

    return data


def generate_multimodal_data(modes, npts):
    nmodes = len(modes)
    d = len(modes[0][0])

    data = np.zeros((nmodes, npts, d)) * np.nan
    for m, mode in enumerate(modes):
        data[m] = np.random.multivariate_normal(mode[0], mode[1], npts)

    if np.sum(np.isnan(data)) > 0:
        raise Exception("NaN found!")

    return data


def get_synthetic_modalities(nstat, nchange, dim, dmin, dmax=None, check_max=False):
    # TODO: seed?
    if dmax is None:
        dmax = 10 * dmin

    stationary_modalities = generate_gaussian_modalities(nmodes=nstat + nchange, dim=dim,
                                                         d0=dmin, dmax=dmax,
                                                         check_max=check_max)
    change_modalities = []
    for i in range(nchange):
        change_modalities.append(stationary_modalities.pop(np.random.choice(len(stationary_modalities))))

    return stationary_modalities, change_modalities

