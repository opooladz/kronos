# Kronos: PSGD — *Kron One-Sided*

A minimal codebase demonstrating Kronos handling 2D layers, with AdamW applied to the rest.

In this example, we show that it's possible to **jointly precondition all layers of the same shape using a shared Q**, and that this approach actually outperforms using a separate Q for each weight matrix.

My theory is that this setup provides **more gradient samples** for performing **Lie group Spectral Stochastic Gradient Descent (SSGD)**. As a result, the whitening preconditioner becomes both more **general** (due to sampling across multiple weights) and more **refined** (due to more SSGD steps).

Enjoy the memory savings—QQᵀ can be cached and reused at all times.


[Imagenet](https://github.com/EliSchwartz/imagenet-sample-images) Samples Needed for code
