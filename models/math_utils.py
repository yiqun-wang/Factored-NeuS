import torch
import numpy as np


def smithG1(cosTheta, alpha):
    sinTheta = torch.sqrt(1.0 - cosTheta * cosTheta)
    tanTheta = sinTheta / (cosTheta + 1e-10)
    root = alpha * tanTheta
    return 2.0 / (1.0 + torch.hypot(root, torch.ones_like(root)))


def l2_normalize(x, eps=torch.tensor(torch.finfo(torch.float32).eps)):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(torch.maximum(torch.sum(x**2, dim=-1, keepdim=True), eps))


def dot(x, y):
    return torch.sum(x * y, dim=-1, keepdim=True)


def reflect(d, n):
    return 2.0 * dot(d, n) * n - d


def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.maximum(np.math.factorial(k), 1e-7)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

    Args:
      l: associated Legendre polynomial degree.
      m: associated Legendre polynomial order.
      k: power of cos(theta).

    Returns:
      A float, the coefficient of the term corresponding to the inputs.
    """
    return ((-1.0)**m * 2.0**l * np.math.factorial(l) / np.maximum(np.math.factorial(k), 1e-7) /
            np.maximum(np.math.factorial(l - k - m), 1e-7) *
            generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    return (np.sqrt(
        (2.0 * l + 1.0) * np.math.factorial(l - m) /
        np.maximum(4.0 * np.pi * np.math.factorial(l + m), 1e-7)) * assoc_legendre_coeff(l, m, k))


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T  # [2,19]
    return ml_array


def generate_ide_fn(deg_view):  # 4
    """Generate integrated directional encoding (IDE) function.

    This function returns a function that computes the integrated directional
    encoding from Equations 6-8 of arxiv.org/abs/2112.03907.

    Args:
      deg_view: number of spherical harmonics degrees to use.

    Returns:
      A function for evaluating integrated directional encoding.

    Raises:
      ValueError: if deg_view is larger than 5.
    """
    if deg_view > 5:
        print('WARNING: Only deg_view of at most 5 is numerically stable.')
    #   raise ValueError('Only deg_view of at most 5 is numerically stable.')

    ml_array = get_ml_array(deg_view)  # [2,19]
    l_max = 2**(deg_view - 1)  # 8

    # Create a matrix corresponding to ml_array holding all coefficients, which,
    # when multiplied (from the right) by the z coordinate Vandermonde matrix,
    # results in the z component of the encoding.
    mat = torch.zeros((l_max + 1, ml_array.shape[1]))  # [9,19]
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).

        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.

        Returns:
          An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.cat([z**i for i in range(mat.shape[0])], dim=-1)  # [n,9]

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.cat(  # [n,19]
            [(x + 1j * y)**m for m in ml_array[0, :]], dim=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, mat)  # [n,19]

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = torch.tensor(0.5 * ml_array[1, :] * (ml_array[1, :] + 1), dtype=torch.float32)  # [19,]
        ide = sph_harms * torch.exp(-sigma * kappa_inv)  # [n,19]

        # Split into real and imaginary parts and return
        res = torch.cat([torch.real(ide), torch.imag(ide)], dim=-1)
        # res[torch.isnan(res)] = 0
        return res

    return integrated_dir_enc_fn


def linear_to_srgb(linear):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    eps = torch.tensor(torch.finfo(torch.float32).eps)
    # eps = torch.tensor(1e-7)
    srgb0 = 323.0 / 25.0 * linear
    srgb1 = (211.0 * torch.maximum(eps, linear) ** (5.0 / 12.0) - 11.0) / 200.0
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb):
    """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    eps = torch.tensor(torch.finfo(torch.float32).eps)
    linear0 = 25.0 / 323.0 * srgb
    linear1 = torch.maximum(eps, ((200.0 * srgb + 11.0) / (211.0))) ** (12.0 / 5.0)
    return torch.where(srgb <= 0.04045, linear0, linear1)


def rgb_to_hsv(x):  # [n,3]
    c_max = torch.max(x, dim=-1, keepdim=True)[0]
    c_min = torch.min(x, dim=-1, keepdim=True)[0]
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:]

    v = c_max  # [n,1]

    v_not0_mask = (v>0)
    v_r_mask = (v==r)
    v_g_mask = (v==g)
    v_b_mask = (v==b)

    s = torch.zeros_like(v)
    s[v_not0_mask] = (v[v_not0_mask] - c_min[v_not0_mask]) / (v[v_not0_mask] + 1e-6)

    h = torch.zeros_like(v)
    h[v_r_mask] = 60.0 * ((g[v_r_mask] - b[v_r_mask]) / (v[v_r_mask] - c_min[v_r_mask] + 1e-6))
    h[v_g_mask] = 120.0 + 60.0 * ((b[v_g_mask] - r[v_g_mask]) / (v[v_g_mask] - c_min[v_g_mask] + 1e-6))
    h[v_b_mask] = 240.0 + 60.0 * ((r[v_b_mask] - g[v_b_mask]) / (v[v_b_mask] - c_min[v_b_mask] + 1e-6))
    
    return h, s, v

