from typing import NamedTuple
import jax
from jax import Array
import jax.numpy as jnp


class LayerParams(NamedTuple):
    attn_norm: Array
    attn_wq: Array
    attn_wkv: Array
    attn_wout: Array
    ffn_norm: Array
    gate: Array
    ffn_w1: Array
    ffn_w2: Array
    ffn_w3: Array


class Params(NamedTuple):
    emb: Array
    layers: LayerParams
    norm: Array
    out: Array


def rope(x: Array, theta: int):
    _, _, t, d = x.shape
    emb = jnp.arange(t)[:, None] / (theta ** (jnp.arange(0, d, 2)[None, :] / d))
    cos, sin = jnp.tile(jnp.cos(emb), 2), jnp.tile(jnp.sin(emb), 2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return jnp.concat((x1, x2), axis=-1) * cos + jnp.concat((-x2, x1), axis=-1) * sin


def rms_norm(x: Array, w: Array, eps: float):
    return w * x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


def attn(x: Array, wq: Array, wkv: Array, wout: Array, theta: int):
    q = jnp.einsum("btc,chd->bhtd", x, wq)
    k, v = jnp.split(jnp.einsum("btc,cgd->bgtd", x, wkv), 2, axis=-1)
    k, v = k.repeat(q.shape[1] // k.shape[1], 1), v.repeat(q.shape[1] // v.shape[1], 1)
    _, _, t, c = q.shape
    q, k = rope(q, theta), rope(k, theta)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(c)
    mask = jnp.triu(jnp.full((t, t), -jnp.inf), k=1)
    values = jnp.einsum("bhtl,bhld->bhtd", jax.nn.softmax(logits + mask, -1), v)
    return jnp.einsum("bhtd,hdc->btc", values, wout)


def swiglu(x: Array, w1: Array, w2: Array, w3: Array):
    return jnp.dot(jax.nn.silu(jnp.dot(x, w1)) * jnp.dot(x, w2), w3)


def ffn(x: Array, gate: Array, w1: Array, w2: Array, w3: Array, topk: int):
    b, t, c = x.shape
    logits, indices = jax.lax.top_k(jnp.einsum("btc,cn->btn", x, gate), topk)
    weights = jax.nn.softmax(logits, -1)
    out = jax.vmap(lambda idx, x: swiglu(x, w1[idx], w2[idx], w3[idx]))(
        indices.reshape(-1),
        x[:, :, None, :].repeat(topk, 2).reshape(-1, c),
    )
    return jnp.sum(out.reshape(b, t, topk, c) * weights[:, :, :, None], 2)


def gpt(x: Array, params: Params, eps: float, theta: int, topk: int):
    x = params.emb[x]

    def f(x: Array, layer: LayerParams):
        h = rms_norm(x, layer.attn_norm, eps)
        x = x + attn(h, layer.attn_wq, layer.attn_wkv, layer.attn_wout, theta)
        h = rms_norm(x, layer.ffn_norm, eps)
        x = x + ffn(h, layer.gate, layer.ffn_w1, layer.ffn_w2, layer.ffn_w3, topk)
        return x, None

    x, _ = jax.lax.scan(f, x, params.layers)
    return rms_norm(x, params.norm, eps) @ params.out


if __name__ == "__main__":
    nv = 50257;d = 384;nb = 6;nh = 6;ng = 2;ne = 8;tk = 2  # noqa: E702
    eps = 1e-5;theta = 10_000  # noqa: E702
    keys = jax.random.split(jax.random.PRNGKey(0), 9)
    init = jax.nn.initializers.xavier_normal()
    params = Params(
        emb=init(keys[0], (nv, d)),
        layers=LayerParams(
            attn_norm=jnp.ones((nb, d)),
            attn_wq=init(keys[1], (nb, d, nh, d // nh)),
            attn_wkv=init(keys[2], (nb, d, ng, 2 * d // nh)),
            attn_wout=init(keys[3], (nb, nh, d // nh, d)),
            ffn_norm=jnp.ones((nb, d)),
            gate=init(keys[4], (nb, d, ne)),
            ffn_w1=init(keys[4], (nb, ne, d, (8 // 3) * d)),
            ffn_w2=init(keys[5], (nb, ne, d, (8 // 3) * d)),
            ffn_w3=init(keys[6], (nb, ne, (8 // 3) * d, d)),
        ),
        norm=jnp.ones((d,)),
        out=init(keys[7], (d, nv)),
    )
    out = gpt(jnp.ones((2, 1024), dtype=int), params, eps, theta, tk)
