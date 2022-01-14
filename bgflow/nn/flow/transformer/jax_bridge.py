try:
    import jax
    import jax.numpy as jnp
    import jax2torch
except:
    jax = None
    jnp = None
    jax2torch = None
import functools


def apply_r(f, g):
    """Function application from right."""
    # (A -> B) (A -> B -> C) => (A -> C)
    return lambda *args, **kwargs: g(f)(*args, **kwargs)


def chain(*fs):
    """Chained function appliction (outer-most first)."""
    return functools.reduce(apply_r, fs[::-1])


def compose2(f, g):
    """Composes two functions."""
    return lambda *args, **kwargs: f(g(*args, **kwargs))


def compose(*fs):
    """Composes functions."""
    return functools.reduce(compose2, fs)


def bisect(bijector, left_bound, right_bound, eps=1e-6):
    """Bisection search."""

    @jax.jit
    def _inverted(target):
        init = (jnp.ones_like(target) * left_bound,
                jnp.ones_like(target) * right_bound)
        n_iters = jnp.ceil(-jnp.log2(eps)).astype(int)

        def _body(_, val):
            left_bound, right_bound = val
            cand = (left_bound + right_bound) / 2
            pred = bijector(cand)
            left_bound = jnp.where(pred < target, cand, left_bound)
            right_bound = jnp.where(pred > target, cand, right_bound)
            return left_bound, right_bound

        return jax.lax.fori_loop(0, n_iters, _body, init)[0]

    return _inverted


def invert(bijector, root_finder):
    """Inverts a bijector with a root finder
       and computes correct gradients using
       implicit differentation."""

    def _forward(outp, cond):
        root = root_finder(lambda x: bijector(x, cond)[0])(outp)
        _, ldj = bijector(root, cond)
        return (root, -ldj), (root, cond)

    def _backward(res, tangents):
        root, cond = res
        root_grad, ldj_grad = tangents

        def _jac_diag(inp):
            outp, vjp_fun = jax.vjp(lambda x: bijector(x, cond)[0], inp)
            return vjp_fun(jnp.ones_like(outp))[0]

        jac_diag = _jac_diag(root)
        root_grad /= jac_diag
        ldj_grad /= jac_diag

        def _log_jac_diag(inp):
            return jnp.log(_jac_diag(inp))

        outp, pullback = jax.vjp(_log_jac_diag, root)
        dldj_dinp = pullback(jnp.ones_like(outp))[0]
        outp_grad = root_grad - dldj_dinp * ldj_grad

        # gradient wrt cond
        def _helper(cond):
            outp, _ = jax.vjp(lambda x: bijector(root, x)[0], cond)
            _, vjp_fun = jax.vjp(lambda x: bijector(x, cond)[0], root)
            jac = vjp_fun(jnp.ones_like(outp))[0]
            return (outp, outp, jac)

        _, pullback = jax.vjp(_helper, cond)
        cond_grad = pullback((
            - root_grad,
            dldj_dinp * ldj_grad,
            - ldj_grad
        ))[0]
        return outp_grad, cond_grad

    @jax.custom_vjp
    def _call(outp, cond):
        return _forward(outp, cond)[0]

    _call.defvjp(_forward, _backward)

    return _call


def with_ldj(bijector):
    """Wraps bijector with automatic log jacobian determinant computation."""
    def _call(x, *other):
        y, vjp_bijector = jax.vjp(lambda x: bijector(x, *other), x)
        ldj = jnp.log(jnp.abs(vjp_bijector(jnp.ones_like(y))[0])).sum()
        return y, ldj
    return _call


def bijector_with_approx_inverse(bijector, domain=None):
    """Wraps bijector with approximate inverse."""
    if domain is None:
        domain = (0, 1)

    root_finder = functools.partial(
        bisect,
        left_bound=domain[0],
        right_bound=domain[1])
    invert_ = functools.partial(
        invert,
        root_finder=root_finder)
    bijector = with_ldj(jax.vmap(bijector))

    def _forward(x, cond):
        return bijector(x, *cond)

    def _inverse(x, cond):
        return invert_(_forward)(x, cond)

    return _forward, _inverse


def to_bgflow(bijector, domain=None):
    """Wraps simple JAX bijector into a transformer,
       that can be used within the bgflow eco-system."""
    return map(compose(
        functools.wraps(bijector),
        jax2torch.jax2torch,
        jax.jit,
        jax.vmap),
        bijector_with_approx_inverse(bijector, domain))
