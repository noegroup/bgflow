
import torch
import torch.utils.dlpack

try:
    import jax
    import jax.numpy as jnp
    import jax.dlpack
except ImportError:
    jax = None
    jnp = None
    jax_dlpack = None
import functools

from .base import Transformer


__all__ = [
    'JaxTransformer',
    'chain',
]


def apply_r(f, g):
    """Function application from right.

       Takes a higher order function g: (A -> B) -> C
       and applies it to the function f: A -> B.

       The result is the fungtion g(f): A -> C.
    """
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

    def _inverted(target):
        init = (jnp.ones_like(target) * left_bound,
                jnp.ones_like(target) * right_bound)
        n_iters = jnp.ceil(-jnp.log2(eps)).astype(int)

        def _body(_, left_right):
            left_bound, right_bound = left_right
            cand = (left_bound + right_bound) / 2
            pred = bijector(cand)
            left_bound = jnp.where(pred < target, cand, left_bound)
            right_bound = jnp.where(pred > target, cand, right_bound)
            return left_bound, right_bound

        return jax.lax.fori_loop(0, n_iters, _body, init)[0]

    return _inverted


def invert_bijector(bijector, root_finder):
    """Inverts a bijector with a root finder
       and computes correct gradients using
       implicit differentation."""

    def _bijector(x, cond):
        return bijector(x, *cond)

    def _forward(outp, *cond):
        root = root_finder(lambda x: _bijector(x, cond)[0])(outp)
        _, ldj = _bijector(root, cond)
        return (root, -ldj), (root, cond)

    def _backward(res, tangents):
        root, cond = res
        root_grad, ldj_grad = tangents

        def _jac_diag(inp):
            outp, vjp_fun = jax.vjp(lambda x: _bijector(x, cond)[0], inp)
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
            outp, _ = jax.vjp(lambda x: _bijector(root, x)[0], cond)
            _, vjp_fun = jax.vjp(lambda x: _bijector(x, cond)[0], root)
            jac = vjp_fun(jnp.ones_like(outp))[0]
            return (outp, outp, jac)

        _, pullback = jax.vjp(_helper, cond)
        cond_grad = pullback((
            - root_grad,
            dldj_dinp * ldj_grad,
            - ldj_grad
        ))[0]
        return (outp_grad, *cond_grad)

    @jax.custom_vjp
    def _call(outp, *cond):
        return _forward(outp, *cond)[0]

    _call.defvjp(_forward, _backward)

    return _call


def with_ldj(bijector):
    """Wraps bijector with automatic log jacobian determinant computation."""
    def _call(x, *params):
        y, vjp_bijector = jax.vjp(lambda x: bijector(x, *params), x)
        ldj = jnp.log(vjp_bijector(jnp.ones_like(y))[0])
        return y, ldj
    return _call


def bijector_with_approx_inverse(bijector, domain=None, eps=1e-8):
    """Wraps bijector with approximate inverse."""
    if domain is None:
        domain = (0, 1)
    root_finder = functools.partial(
        bisect,
        left_bound=domain[0],
        right_bound=domain[1],
        eps=eps)
    invert = functools.partial(
        invert_bijector,
        root_finder=root_finder)
    forward = with_ldj(bijector)
    inverse = invert(forward)
    return forward, inverse


def flip(fn, permutation=(1, 0)):
    """Flips argument order of function according to permutation."""
    def inner(*args, **kwargs):
        args = tuple(args[p] for p in permutation) + args[len(permutation):]
        return fn(*args, **kwargs)
    return inner


def map_if(predicate):
    def wrap(fn):
        def inner(x):
            if predicate(x):
                return fn(x)
            else:
                return x
        return inner
    return wrap


@functools.wraps(functools.reduce)
def tree_reduce(*args):
    assert len(args) >= 2
    args = list(args)
    args[1] = jax.tree_util.tree_flatten(args[1])[0]
    return functools.reduce(*args)


def assert_contiguous(x):
    return x.contiguous()


is_torch_tensor = functools.partial(flip(isinstance), torch.Tensor)
if jnp is not None:
    is_jax_ndarray = functools.partial(flip(isinstance), jnp.ndarray)
if jax is not None and jax.dlpack is not None:
    to_torch_tensor = compose(torch.utils.dlpack.from_dlpack, jax.dlpack.to_dlpack)
    to_jax_ndarray = compose(jax.dlpack.from_dlpack, torch.utils.dlpack.to_dlpack, assert_contiguous)


class JaxWrapper(torch.autograd.Function):

        @staticmethod
        def forward(ctx, fn, *args):
            args = jax.tree_util.tree_map(map_if(is_torch_tensor)(to_jax_ndarray), args)
            result, ctx.fun_vjp = jax.vjp(fn, *args)
            result_flat, result_tree = jax.tree_util.tree_flatten(result)
            ctx.result_tree = result_tree
            return (*jax.tree_util.tree_map(map_if(is_jax_ndarray)(to_torch_tensor), result_flat),
                    result_tree)

        @staticmethod
        def backward(ctx, *tangents):
            tangents = jax.tree_util.tree_map(map_if(is_torch_tensor)(to_jax_ndarray), tangents)
            tangents = jax.jax.tree_util.tree_unflatten(ctx.result_tree, tangents[:-1])
            grads = ctx.fun_vjp(tangents)
            return (None, *jax.tree_util.tree_flatten(jax.tree_util.tree_map(map_if(is_jax_ndarray)(to_torch_tensor), grads))[0])


def wrap_jax_fun(fn):
    @functools.wraps(fn)
    def inner(*args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        *result_flat, result_tree = JaxWrapper.apply(fn, *args_flat)
        return jax.tree_util.tree_unflatten(result_tree, result_flat)
    return inner


def assert_float32(x):
    if x.dtype != torch.float32:
        raise ValueError(f'dtype {x.dtype} currently not supported by jax bridge')
        # if we want to support this, enable x64 globally during compile via a context mgr such as:
        # >>> from jax.config import config
        # >>> config.update("jax_enable_x64", True)
        # >>> yield
        # >>> config.update("jax_enable_x64", False)


def nested_vmap(fn, indices):
    """Applies nested vmap for specified vectorization indices."""
    for idx in indices:
        fn = jax.vmap(fn, in_axes=idx, out_axes=idx)
    return fn


def jax_compile(bijector, vmap_indices, backend, domain=None, bisection_eps=1e-8):
    """Wraps simple JAX bijector into a transformer,
       that can be used within the bgflow eco-system."""
    compile_bijector = compose(functools.partial(jax.jit))
    fwd, bwd = bijector_with_approx_inverse(nested_vmap(bijector, vmap_indices), domain, bisection_eps)
    return tuple(map(compile_bijector, (fwd, bwd)))


def torch_to_jax_backend(backend):
    """Assert correct backend naming."""
    if backend == 'cuda':
        backend = 'gpu'
    return backend


def to_torch_impl_(bijector, vmap_indices, backend, domain=None, bisection_eps=1e-8):
    """Helper impl function that can be cashed according
       to `vmap_indices` and `backend`"""
    fwd, bwd = jax_compile(bijector, vmap_indices, backend, domain, bisection_eps)
    return tuple(map(wrap_jax_fun, (fwd, bwd)))


def to_torch(bijector, vmap_indices=None, domain=None, bisection_eps=1e-8):
    """Converts a simple JAX bijector into a torch bijector with
        - numerical inverses
        - automatic computation of log det jac

       `vmap_indices`: Specify axes which vmap is applied on.
                       If set to None applies vmap over the full tensor."""
    cached_compile = functools.lru_cache(functools.partial(to_torch_impl_, bijector))

    def _cached(x):
        indices = vmap_indices
        if indices is None:
            indices = tuple(range(len(x.shape)))
        backend = torch_to_jax_backend(x.device.type)
        return cached_compile(indices, backend, domain, bisection_eps)

    def _fwd(x, *params):
        assert_float32(x)
        fwd, _ = _cached(x)
        return fwd(x, *params)

    def _bwd(x, *params):
        assert_float32(x)
        _, bwd = _cached(x)
        return bwd(x, *params)

    return _fwd, _bwd


class JaxTransformer(Transformer):
    """Simple wrapper to make bijectors usable in coupling
       layers of bgflow.

       bijector: JAX bijector
       compute_params: function producing params for the
                       bijector."""

    def __init__(self, bijector, compute_params, reduce_jacobian=True,
                 domain=None, bisection_eps=1e-8):
        super().__init__()
        self._compute_params = compute_params
        fwd, bwd = to_torch(bijector)
        self.fwd = fwd
        self.bwd = bwd
        self.reduce_jacobian = reduce_jacobian

    def _forward(self, x, y, *args, **kwargs):
        params = self._compute_params(x, y.shape, *args, **kwargs)
        newy, ldj = self.fwd(y, *params)
        if self.reduce_jacobian:
            ldj = ldj.sum(dim=-1, keepdim=True)
        return newy, ldj

    def _inverse(self, x, y, *args, **kwargs):
        params = self._compute_params(x, y.shape, *args, **kwargs)
        newy, ldj = self.bwd(y, *params)
        if self.reduce_jacobian:
            ldj = ldj.sum(dim=-1, keepdim=True)
        return newy, ldj
