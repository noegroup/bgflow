import torch
import warnings

from bgflow.utils.autograd import brute_force_jacobian


# permutations used in the xyz->ic matrix determinant computation
_INIT_ICS_DET_PERMUTATIONS = torch.LongTensor([
    (0, 1, 2, 3, 6, 7, 4, 5, 8),
    (0, 1, 2, 3, 6, 8, 4, 5, 7),
    (0, 1, 2, 3, 7, 6, 4, 5, 8),
    (0, 1, 2, 3, 7, 8, 4, 5, 6),
    (0, 1, 2, 3, 8, 6, 4, 5, 7),
    (0, 1, 2, 3, 8, 7, 4, 5, 6),
    (0, 1, 2, 4, 6, 7, 3, 5, 8),
    (0, 1, 2, 4, 6, 8, 3, 5, 7),
    (0, 1, 2, 4, 7, 6, 3, 5, 8),
    (0, 1, 2, 4, 7, 8, 3, 5, 6),
    (0, 1, 2, 4, 8, 6, 3, 5, 7),
    (0, 1, 2, 4, 8, 7, 3, 5, 6),
    (0, 1, 2, 5, 6, 7, 3, 4, 8),
    (0, 1, 2, 5, 6, 7, 4, 3, 8),
    (0, 1, 2, 5, 6, 8, 3, 4, 7),
    (0, 1, 2, 5, 6, 8, 4, 3, 7),
    (0, 1, 2, 5, 7, 6, 3, 4, 8),
    (0, 1, 2, 5, 7, 6, 4, 3, 8),
    (0, 1, 2, 5, 7, 8, 3, 4, 6),
    (0, 1, 2, 5, 7, 8, 4, 3, 6),
    (0, 1, 2, 5, 8, 6, 3, 4, 7),
    (0, 1, 2, 5, 8, 6, 4, 3, 7),
    (0, 1, 2, 5, 8, 7, 3, 4, 6),
    (0, 1, 2, 5, 8, 7, 4, 3, 6)
])


# signs used in the xyz->ic matrix determinant computation
_INIT_ICS_DET_PERMUTATION_SIGNS = torch.FloatTensor([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1])


# permutations used in the ic->xyz matrix determinant computation
_INIT_XYZ_DET_PERMUTATIONS = torch.LongTensor([
    (0, 1, 2, 3, 6, 7, 4, 5, 8),
    (0, 1, 2, 3, 6, 7, 4, 8, 5),
    (0, 1, 2, 3, 6, 7, 5, 4, 8),
    (0, 1, 2, 3, 6, 7, 5, 8, 4),
    (0, 1, 2, 3, 6, 7, 8, 4, 5),
    (0, 1, 2, 3, 6, 7, 8, 5, 4),
    (0, 1, 2, 6, 3, 7, 4, 5, 8),
    (0, 1, 2, 6, 3, 7, 4, 8, 5),
    (0, 1, 2, 6, 3, 7, 5, 4, 8),
    (0, 1, 2, 6, 3, 7, 5, 8, 4),
    (0, 1, 2, 6, 3, 7, 8, 4, 5),
    (0, 1, 2, 6, 3, 7, 8, 5, 4),
    (0, 1, 2, 6, 7, 3, 4, 5, 8),
    (0, 1, 2, 6, 7, 3, 4, 8, 5),
    (0, 1, 2, 6, 7, 3, 5, 4, 8),
    (0, 1, 2, 6, 7, 3, 5, 8, 4),
    (0, 1, 2, 6, 7, 3, 8, 4, 5),
    (0, 1, 2, 6, 7, 3, 8, 5, 4),
    (0, 1, 2, 7, 6, 3, 4, 5, 8),
    (0, 1, 2, 7, 6, 3, 4, 8, 5),
    (0, 1, 2, 7, 6, 3, 5, 4, 8),
    (0, 1, 2, 7, 6, 3, 5, 8, 4),
    (0, 1, 2, 7, 6, 3, 8, 4, 5),
    (0, 1, 2, 7, 6, 3, 8, 5, 4)
])


# signs used in the ic->xyz matrix determinant computation
_INIT_XYZ_DET_PERMUTATION_SIGNS = torch.FloatTensor([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])



def outer(x, y):
    """ outer product between input tensors """
    return x[..., None] @ y[..., None, :]


def skew(x):
    """
        returns skew symmetric 3x3 form of a 3 dim vector
    """
    assert len(x.shape) > 1, "`x` requires at least 2 dimensions"
    zero = torch.zeros(*x.shape[:-1]).to(x)
    a, b, c = x[..., 0], x[..., 1], x[..., 2]
    s = torch.stack(
        [
            torch.stack([zero, c, -b], dim=-1),
            torch.stack([-c, zero, a], dim=-1),
            torch.stack([b, -a, zero], dim=-1),
        ],
        dim=-1,
    )
    return s


def det2x2(a):
    """ batch determinant of a 2x2 matrix """
    return a[..., 0, 0] * a[..., 1, 1] - a[..., 1, 0] * a[..., 0, 1]


def det3x3(a):
    """ batch determinant of a 3x3 matrix """
    return (torch.cross(a[..., 0, :], a[..., 1, :], dim=-1) * a[..., 2, :]).sum(dim=-1)


def tripod(p1, p2, p3, eps=1e-7, raise_warnings=True, enforce_boundaries=True):
    """ computes a unique orthogonal basis for input points """
    e1 = p2 - p1
    e1_norm = torch.norm(e1, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(e1_norm < eps):
            warnings.warn("singular division in computing orthogonal basis")
    if enforce_boundaries:
        e1_norm = e1_norm.clamp_min(eps)

    e1 = e1 / e1_norm
    u = p3 - p1
    e2 = torch.cross(u, e1)
    e2_norm = torch.norm(e2, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(e2_norm < eps):
            warnings.warn("singular division in computing orthogonal basis")
    if enforce_boundaries:
        e2_norm = e2_norm.clamp_min(eps)

    e2 = e2 / e2_norm
    e3 = torch.cross(e2, e1)
    return -e3, -e2, e1


def orientation(p1, p2, p3, eps=1e-7, raise_warnings=True, enforce_boundaries=True):
    """ computes unique orthogonal basis transform for input points """
    return torch.stack(tripod(p1, p2, p3, eps, raise_warnings, enforce_boundaries), dim=-1)


def dist_deriv(x1, x2, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """
        computes distance between input points together with
        the Jacobian wrt to `x1`
    """
    r = x2 - x1
    rnorm = torch.norm(r, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(rnorm < eps):
            warnings.warn("singular division in distance computation")
    if enforce_boundaries:
        rnorm = rnorm.clamp_min(eps)

    dist = rnorm[..., 0]
    J = -r / rnorm
    # J = _safe_div(-r, rnorm)
    return dist, J


def angle_deriv(x1, x2, x3, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """
        computes angle between input points together with
        the Jacobian wrt to `x1`
    """
    r12 = x1 - x2
    r12_norm = torch.norm(r12, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(r12_norm < eps):
            warnings.warn("singular division in angle computation")
    if enforce_boundaries:
        r12_norm = r12_norm.clamp_min(eps)

    rn12 = r12 / r12_norm

    J = (torch.eye(3).to(x1) - outer(rn12, rn12)) / r12_norm[..., None]

    r32 = x3 - x2
    r32_norm = torch.norm(r32, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(r32_norm < eps):
            warnings.warn("singular division in angle computation")
    if enforce_boundaries:
        r32_norm = r32_norm.clamp_min(eps)

    rn32 = r32 / r32_norm

    cos_angle = torch.sum(rn12 * rn32, dim=-1)
    J = rn32[..., None, :] @ J

    if raise_warnings:
        if torch.any((cos_angle < -1. + eps) & (cos_angle > 1. - eps)):
            warnings.warn("singular radians in angle computation")
    if enforce_boundaries:
        cos_angle = cos_angle.clamp(-1. + eps, 1. - eps)

    a = torch.acos(cos_angle)

    J = -J / torch.sqrt(1.0 - cos_angle.pow(2)[..., None, None])

    return a, J[..., 0, :]


def torsion_deriv(x1, x2, x3, x4, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """
        computes torsion angle between input points together with
        the Jacobian wrt to `x1`.
    """
    b0 = -1.0 * (x2 - x1)

    # TODO not used can be removed in next refactor
    # db0_dx1 = torch.eye(3).to(x1)

    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1norm = torch.norm(b1, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(b1norm < eps):
            warnings.warn("singular division in distance computation")
    if enforce_boundaries:
        b1norm = b1norm.clamp_min(eps)

    b1_normalized = b1 / b1norm

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    #
    # dv_db0 = jacobian of v wrt b0
    v = b0 - torch.sum(b0 * b1_normalized, dim=-1, keepdim=True) * b1_normalized
    dv_db0 = torch.eye(3)[None, None, :, :].to(x1) - outer(b1_normalized, b1_normalized)

    w = b2 - torch.sum(b2 * b1_normalized, dim=-1, keepdim=True) * b1_normalized

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    #
    # dx_dv = jacobian of x wrt v
    x = torch.sum(v * w, dim=-1, keepdim=True)
    dx_dv = w[..., None, :]

    # b1xv = fast cross product between b1_normalized and v
    # given by multiplying v with the skew of b1_normalized
    # (see https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product)
    #
    # db1xv_dv = Jacobian of b1xv wrt v
    A = skew(b1_normalized)
    b1xv = (A @ (v[..., None]))[..., 0]
    db1xv_dv = A

    # y = dot product of b1xv and w
    # dy_db1xv = Jacobian of v wrt b1xv
    y = torch.sum(b1xv * w, dim=-1, keepdim=True)
    dy_db1xv = w[..., None, :]

    x = x[..., None]
    y = y[..., None]

    # a = torsion angle spanned by unit vector (x, y)
    # xysq = squared norm of (x, y)
    # da_dx = Jacobian of a wrt xysq
    a = torch.atan2(y, x)
    xysq = x.pow(2) + y.pow(2)

    if raise_warnings:
        if torch.any(xysq < eps):
            warnings.warn("singular division in torsion computation")
    if enforce_boundaries:
        xysq = xysq.clamp_min(eps)

    da_dx = -y / xysq
    da_dy = x / xysq

    # compute derivative with chain rule
    J = da_dx @ dx_dv @ dv_db0 + da_dy @ dy_db1xv @ db1xv_dv @ dv_db0

    return a[..., 0, 0], J[..., 0, :]


def _permutation_parity(perm):
    """computes parity of permutation in O(n log n) time using cycle decomposition"""
    n = len(perm)
    not_visited = set(range(n))
    i = perm[0]
    c = 1
    while len(not_visited) > 0:
        if i in not_visited:
            not_visited.remove(i)      
        else:       
            c += 1
            i = not_visited.pop()
        i = perm[i]            
    return (-1) ** (n - c)


def _determinant_from_permutations(mat, permutations, signs):
    """ The determinant of a NxN matrix A is 
            
            \det A = \sum_{\sigma} \sign{\sigma} \prod_{i} A[i, \sigma[i]]
        
        where the sum is over all possible permutation \sigma of {1, ..., N}
        
        If we know the non-vanishing permutations and their signs, we can compute it explicitly.
        
        The function takes a matrix, all non-vanishing permutations with corresponding signs and
        uses it to compute the determinant of it.
    """
    n = mat.shape[-1]
    return (mat[..., torch.arange(n, device=mat.device), permutations].prod(dim=-1) * signs).sum(dim=-1)
    
    
def _to_euler_angles(x, y, z):
    """ converts a basis made of three orthonormal vectors into the corresponding proper x-y-z euler angles 
        output values are in [0, pi]
    """
    x, y, z = basis.chunk
    alpha = torch.acos(- z[..., 1] / (1. - z[..., 2].pow(2)).sqrt())
    beta = torch.acos(z[..., 2])
    gamma = torch.acos(y[..., 2] / (1. - z[..., 2].pow(2)).sqrt())
    return alpha, beta, gamma


def _rotmat3x3(theta, axis):
    """ computes the matrix corresponding to a 2D rotation around `axis` in 3D """
    r = torch.eye(3, dtype=theta.dtype, device=theta.device).repeat(*theta.shape[:-1], 1, 1)    
    axes = [i for i in range(3) if i != axis]
    r[..., axes[0], axes[0]] = torch.cos(theta)
    r[..., axes[0], axes[1]] = torch.sin(theta)
    r[..., axes[1], axes[0]] = -torch.sin(theta)
    r[..., axes[1], axes[1]] = torch.cos(theta)
    return r


def _from_euler_angles(alpha, beta, gamma):
    """ converts proper euler angles in x-y-z representation into the corresponding rotation matrix
        input values are in [0, pi]
    """
    xrot = _rotmat3x3(alpha, axis=2)
    yrot = _rotmat3x3(beta, axis=0)
    zrot = _rotmat3x3(gamma, axis=2)
    return xrot @ yrot @ zrot


def init_ics2xyz(x0, d01, d12, a012, alpha, beta, gamma, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """ computes the first three points given initial ICs and the position of the first point 
        
        Parameters:
        -----------
        x0: first point
        d01: distance between x0 and x1
        d12: distance between x1 and x2
        a021: angle between (x2 - x0) and (x1 - x0)
        alpha: first euler angle (in [0, pi])
        beta: second euler angle (in [0, pi])
        gamma: third euler angle (in [0, pi])
        
        Returns:
        x0: first point
        x1: second point
        x2: third point
        dlogp: density change
    """
    
     # enable grad to use autograd for jacobian computation
    with torch.enable_grad():
        
        # needed for autograd backward pass        
        # we flatten the x0, the ics and the euler angles into a 9-dimensional state vector
        xs = torch.cat([x0.squeeze(-2), d01, d12, a012, alpha, beta, gamma], dim=-1).requires_grad_(True)
        x0 = xs[..., :3].unsqueeze(-2)
        d01, d12, a012, alpha, beta, gamma = xs[..., 3:].chunk(6, dim=-1)
        
        n_batch = d01.shape[0]
        dlogp = 0

        # first point placed in origin
        p0 = torch.zeros(n_batch, 1, 3, device=d01.device, dtype=d01.dtype)

        # second point placed in z-axis
        p1 = torch.zeros_like(x0)
        p1[..., 2] = d01

        # third point placed wrt to p0 and p1
        p2, J2 = ic2xy0_deriv(
            p1,
            p0,
            d12[:, None],
            a012[:, None],
            eps=eps,
            enforce_boundaries=enforce_boundaries,
            raise_warnings=raise_warnings
        )

        # compute rotation matrix from euler angles
        R = _from_euler_angles(alpha, beta, gamma)
    
        # bring back to original reference frame
        x1 = torch.einsum("bnd, bed -> bne", p1, R) + x0
        x2 = torch.einsum("bnd, bed -> bne", p2, R) + x0
        
        # now flatten the three output points into a 9 dimensional state vector
        ys = torch.cat([x0.squeeze(-2), x1.squeeze(-2), x2.squeeze(-2)], dim=-1)
        
        # compute the 9x9 jacobian using bruteforce autograd
        J = brute_force_jacobian(ys, xs)
        
        # we can compute the determinant of this jacobian
        # by summing only the 24 non-vanishing permuations
        dlogp = _determinant_from_permutations(
            J,
            _INIT_XYZ_DET_PERMUTATIONS,
            _INIT_XYZ_DET_PERMUTATION_SIGNS.to(J)
        ).abs().log().unsqueeze(-1)

        return x0, x1, x2, dlogp

    
def init_xyz2ics(x0, x1, x2, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """ computes the initial ICs and the position of the first poin given first three points
        
        Parameters:
        -----------        
        x0: first point
        x1: second point
        x2: third point
        
        Returns:
        --------
        x0: first point
        d01: distance between x0 and x1
        d12: distance between x1 and x2
        a021: angle between (x2 - x0) and (x1 - x0)
        alpha: first euler angle (in [0, pi])
        beta: second euler angle (in [0, pi])
        gamma: third euler angle (in [0, pi])
        dlogp: density change
    """
    
    # enable grad to use autograd for jacobian computation
    with torch.enable_grad():
        
        # needed for autograd backward pass        
        # we flatten the three input into a 9-dimensional state vector
        xs = torch.cat([x0, x1, x2], dim=-1).requires_grad_(True)
        x0, x1, x2 = xs.chunk(3, dim=-1)
        
        # compute ICs as usual
        d01, _ = dist_deriv(
            x0, x1,
            eps=eps, 
            enforce_boundaries=enforce_boundaries, 
            raise_warnings=raise_warnings
        )
        d12, _ = dist_deriv(x1, x2,
            eps=eps, 
            enforce_boundaries=enforce_boundaries, 
            raise_warnings=raise_warnings
        )
        a012, _ = angle_deriv(x0, x1, x2,
            eps=eps, 
            enforce_boundaries=enforce_boundaries, 
            raise_warnings=raise_warnings
        )

        # build a basis made of the first three points
        basis = tripod(x0, x1, x2,
            eps=eps, 
            enforce_boundaries=enforce_boundaries, 
            raise_warnings=raise_warnings
        )
        
        # and compute the euler angles given this basis (range is [0, pi])
        alpha, beta, gamma = _to_euler_angles(*basis)
        
        # now we flatten the outputs (x0, ics, euler angles) into a 9-dim output vec
        ys = torch.cat([x0.squeeze(-2), d01, d12, a012, alpha, beta, gamma], dim=-1)        
        
        # compute the 9x9 jacobian via autograd
        J = brute_force_jacobian(ys, xs)
        
        # we can compute the determinant of this jacobian
        # by summing only the 24 non-vanishing permuations
        dlogp = _determinant_from_permutations(
            J,
            _INIT_ICS_DET_PERMUTATIONS,
            _INIT_ICS_DET_PERMUTATION_SIGNS.to(J)
        ).abs().log().unsqueeze(-1)
    
    return x0, d01, d12, a012, alpha, beta, gamma, dlogp
