from nutils import function, evaluable, _pyast
from functools import partial
import numpy as np


def _construct_res(p, pc, q, λ, h, g):
    e = np.exp(-h * λ * (2 * p - pc))
    return np.array(
        [
            p + λ * (2 * p - pc),
            pc * e,
            (1 + g * λ) * q,
            p * (p - pc) + q**2,
        ]
    )


def _construct_jac(p, pc, q, λ, h, g):
    e = np.exp(-h * λ * (2 * p - pc))
    return np.array(
        [
            [1 + 2 * λ, -λ, 0, 2 * p - pc],
            [pc * e * -2 * h * λ, e + pc * e * h * λ, 0, pc * e * -h * (2 * p - pc)],
            [0, 0, 1 + g * λ, g * q],
            [2 * p - pc, -p, 2 * q, 0],
        ]
    )


def _returnmap(p0, pc0, q0, h, g, tol):
    v0 = np.stack([p0, pc0, q0, 0.0])
    if pc0 == 0 or p0 * (p0 - pc0) + q0**2 <= tol:  # elastic
        return v0
    v = v0.copy()
    for _ in range(9):
        res = _construct_res(*v, h, g) - v0
        if np.linalg.norm(res) < tol:
            return v
        jac = _construct_jac(*v, h, g)
        v -= np.linalg.solve(jac, res)
    raise Exception("return map failed to converge in 9 iterations")


def _invjac(v, h, g):
    if v[3] == 0:  # elastic
        return np.eye(4)
    jac = _construct_jac(*v, h, g)
    return np.linalg.inv(jac)


class _MCC_ReturnMap(evaluable.Array):
    p0: evaluable.Array
    pc0: evaluable.Array
    q0: evaluable.Array
    h: evaluable.Array
    g: evaluable.Array
    tol: float

    def __post_init__(self):
        self.shape = (*self.p0.shape, evaluable.constant(4))
        self.dtype = float
        self.dependencies = self.p0, self.pc0, self.q0, self.h, self.g

    def _compile(self, builder):
        args = builder.compile(self.dependencies)
        evalf = _pyast.Variable("_returnmap")
        builder._globals[evalf.py_expr] = np.vectorize(
            _returnmap, excluded={"tol"}, signature="(),(),(),(),(),()->(n)"
        )
        out = builder.get_variable_for_evaluable(self)
        builder.get_block_for_evaluable(self).assign_to(
            out, evalf.call(*args, _pyast.LiteralFloat(self.tol))
        )
        return out

    def _derivative(self, var, seen):
        invjac = _MCC_InvJac(self)
        return (
            evaluable.einsum(
                "Ai,AB->AiB", invjac[..., 0], self.p0._derivative(var, seen)
            )
            + evaluable.einsum(
                "Ai,AB->AiB", invjac[..., 1], self.pc0._derivative(var, seen)
            )
            + evaluable.einsum(
                "Ai,AB->AiB", invjac[..., 2], self.q0._derivative(var, seen)
            )
        )
        # TODO add derivative for h, g


class _MCC_InvJac(evaluable.Array):
    rm: _MCC_ReturnMap

    def __post_init__(self):
        self.shape = (*self.rm.shape, self.rm.shape[-1])
        self.dtype = float
        self.dependencies = (self.rm,)

    def _compile(self, builder):
        args = builder.compile((self.rm, self.rm.h, self.rm.g))
        evalf = _pyast.Variable("_invjac")
        builder._globals[evalf.py_expr] = np.vectorize(
            _invjac, signature="(n),(),()->(n,n)"
        )
        out = builder.get_variable_for_evaluable(self)
        builder.get_block_for_evaluable(self).assign_to(out, evalf.call(*args))
        return out


def hooke(εe, K, G, Δp):
    I = function.eye(3)
    εev = np.trace(εe)
    εed = εe - (εev / 3) * I
    p = K * εev + Δp
    s = (2 * G) * εed
    q = np.sqrt(1.5 * np.sum(s**2))
    σ = p * I + s
    return σ, p, q, s


def mcc(ε, K, G, Δp, εp, M, h, pcinit, tol):
    σ0, p0, q0, s0 = hooke(ε - εp, K, G, Δp)
    pc0 = pcinit * np.exp(h * np.trace(εp))
    g = 6 * G * K**-1 * M**-2
    p_K, pc_K, q_MK, λK = function._Wrapper(
        partial(_MCC_ReturnMap, tol=tol),
        function.asarray(p0 / K),
        function.asarray(pc0 / K),
        function.asarray(q0 / (M * K)),
        function.asarray(h),
        function.asarray(g),
        shape=(4,),
        dtype=float,
    )
    I = function.eye(3)
    p = p_K * K
    pc = pc_K * K
    q = q_MK * (M * K)
    s = s0 / (1 + g * λK)
    σ = p * I + s
    Δεp = λK * ((1 / 3) * (2 * p_K - pc_K) * I + 3 * K**-1 * M**-2 * s)
    return σ, p, q, s, pc, εp + Δεp
