from typing import NamedTuple
import jax
import jax.numpy as jnp


class LayerParams(NamedTuple):
    attn_norm: jax.Array
    attn_wq: jax.Array
    attn_wkv: jax.Array
    attn_wout: jax.Array
    ffn_norm: jax.Array
    ffn_w1: jax.Array
    ffn_w2: jax.Array
    ffn_w3: jax.Array


class Params(NamedTuple):
    emb: jax.Array
    layers: LayerParams
    norm: jax.Array
    out: jax.Array


def rope(x: jax.Array, theta: int):
    _, _, t, d = x.shape
    emb = jnp.arange(t)[:, None] / (theta ** (jnp.arange(0, d, 2)[None, :] / d))
    cos, sin = jnp.tile(jnp.cos(emb), 2), jnp.tile(jnp.sin(emb), 2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return jnp.concat((x1, x2), axis=-1) * cos + jnp.concat((-x2, x1), axis=-1) * sin


def rms_norm(x: jax.Array, w: jax.Array, eps: float):
    return w * x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


def attn(x: jax.Array, wq: jax.Array, wkv: jax.Array, wout: jax.Array, theta: int):
    q = jnp.einsum("btc,chd->bhtd", x, wq)
    k, v = jnp.split(jnp.einsum("btc,cgd->bgtd", x, wkv), 2, axis=-1)
    k, v = k.repeat(q.shape[1] // k.shape[1], 1), v.repeat(q.shape[1] // v.shape[1], 1)
    _, _, t, c = q.shape
    q, k = rope(q, theta), rope(k, theta)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(c)
    mask = jnp.triu(jnp.full((t, t), -jnp.inf), k=1)
    values = jnp.einsum("bhtl,bhld->bhtd", jax.nn.softmax(logits + mask, -1), v)
    return jnp.einsum("bhtd,hdc->btc", values, wout)


def ffn(x: jax.Array, w1: jax.Array, w2: jax.Array, w3: jax.Array):
    return jnp.dot(jax.nn.silu(jnp.dot(x, w1)) * jnp.dot(x, w2), w3)


def gpt(x: jax.Array, params: Params, eps: float, theta: int):
    x = params.emb[x]

    def f(x: jax.Array, layer: LayerParams):
        h = rms_norm(x, layer.attn_norm, eps)
        x = x + attn(h, layer.attn_wq, layer.attn_wkv, layer.attn_wout, theta)
        h = rms_norm(x, layer.ffn_norm, eps)
        return x + ffn(h, layer.ffn_w1, layer.ffn_w2, layer.ffn_w3), None

    x, _ = jax.lax.scan(f, x, params.layers)
    return rms_norm(x, params.norm, eps) @ params.out


if __name__ == "__main__":
    nv = 50257;d = 384;ctx_len = 1024;nb = 6;nh = 6;ng = 2  # noqa: E702
    eps = 1e-5;theta = 10_000  # noqa: E702
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    init = jax.nn.initializers.xavier_normal()
    params = Params(
        emb=init(keys[0], (nv, d)),
        layers=LayerParams(
            attn_norm=jnp.ones((nb, d)),
            attn_wq=init(keys[1], (nb, d, nh, d // nh)),
            attn_wkv=init(keys[2], (nb, d, ng, 2 * d // nh)),
            attn_wout=init(keys[3], (nb, nh, d // nh, d)),
            ffn_norm=jnp.ones((nb, d)),
            ffn_w1=init(keys[4], (nb, d, (8 // 3) * d)),
            ffn_w2=init(keys[5], (nb, d, (8 // 3) * d)),
            ffn_w3=init(keys[6], (nb, (8 // 3) * d, d)),
        ),
        norm=jnp.ones((d,)),
        out=init(keys[7], (d, nv)),
    )
    out = gpt(jnp.ones((2, 1024), dtype=int), params, eps, theta)
