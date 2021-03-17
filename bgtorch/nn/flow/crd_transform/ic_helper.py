import torch
import warnings

# TODO numpy-style docstrings for single functions


# def _safe_div(x, y):
# """ guarantees x/y -> 0 if x -> 0 and y -> 0 """
# return torch.where(x.abs() > 0.0, x / y, torch.zeros_like(x))


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


def tripod(p1, p2, p3):
    """ computes a unique orthogonal basis for input points """
    e1 = p2 - p1
    e1 = e1 / torch.norm(e1, dim=-1, keepdim=True)
    u = p3 - p1
    e2 = torch.cross(u, e1)
    e2 = e2 / torch.norm(e2, dim=-1, keepdim=True)
    e3 = torch.cross(e2, e1)
    return -e3, -e2, e1


def orientation(p1, p2, p3):
    """ computes unique orthogonal basis transform for input points """
    return torch.stack(tripod(p1, p2, p3), dim=-1)


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
    rn12 = r12 / r12_norm
    # rn12 = _safe_div(r12, r12_norm)

    # TODO replaced for safe division - remove in factoring
    J = (torch.eye(3).to(x1) - outer(rn12, rn12)) / r12_norm[..., None]
    # J = _safe_div((torch.eye(3).to(x1) - outer(rn12, rn12)), r12_norm[..., None])

    r32 = x3 - x2
    r32_norm = torch.norm(r32, dim=-1, keepdim=True)
    # TODO replaced for safe division - remove in factoring
    rn32 = r32 / r32_norm
    # rn32 = _safe_div(r32, r32_norm)

    cos_angle = torch.sum(rn12 * rn32, dim=-1)
    J = rn32[..., None, :] @ J

    if raise_warnings:
        if torch.any((cos_angle < -1. + eps) & (cos_angle > 1. - eps)):
            warnings.warn("singular radians in angle computation")
    if enforce_boundaries:
        cos_angle = cos_angle.clamp(-1. + eps, 1. - eps)

    a = torch.acos(cos_angle)

    # TODO replaced for safe division - remove in factoring
    J = -J / torch.sqrt(1.0 - cos_angle.pow(2)[..., None, None])
    # J = _safe_div(-J, torch.sqrt(1.0 - cos_angle.pow(2)[..., None, None]))

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
    # TODO replaced for safe division - remove in factoring
    b1_normalized = b1 / b1norm
    # b1_normalized = _safe_div(b1, b1norm)

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
