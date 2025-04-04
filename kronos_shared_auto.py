import torch
import os

def safe_solve_triangular(A, B, upper=True, left=True):
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    result = torch.linalg.solve_triangular(A, B, upper=upper, left=left)
    return result.to(torch.bfloat16)

def single_sided_whitening(G, Q, lr_param=0.5):
    G = G.to(torch.bfloat16)
    Q = Q.to(torch.bfloat16)

    m, n = G.shape
    assert m >= n

    V = torch.randn_like(G) / m**0.5
    A = G @ Q.T
    Bh = safe_solve_triangular(Q, V, upper=True, left=False)
    AhA = A.T @ A
    BBh = Bh.T @ Bh
    Q = Q - lr_param / norm_lower_bound(AhA + BBh) * torch.triu(AhA - BBh) @ Q
    return Q.to(torch.float32)

def _lb(A: Tensor, max_abs: Tensor):
    """Cheap lower bound for the spectral norm of A."""
    A /= max_abs
    a0 = torch.einsum("ij,ij->j", A, A)
    i = torch.argmax(a0)
    x = torch.index_select(A, 1, i).flatten().contiguous()
    x = torch.einsum("i,ij->j", x, A)
    x /= x.norm()
    x = torch.einsum("j,kj->k", x, A)
    x = x.norm()
    x *= max_abs
    return x

def norm_lower_bound(A):
    max_abs = A.norm(float("inf"))
    return torch.where(max_abs > 0, _lb(A, max_abs), max_abs)

class Kronos_sharedQ(torch.optim.Optimizer):
    def __init__(self, kronos_params, lr=3e-4, lr_param=0.1, momentum=0.95, nesterov=True,
                 whitening_prob=1.0,
                 adamw_params=None, adamw_lr=3e-4, adamw_betas=(0.95, 0.95), adamw_eps=1e-8, adamw_wd=0):

        defaults = dict(lr=lr, lr_param=lr_param, momentum=momentum, nesterov=nesterov,
                        whitening_prob=whitening_prob,
                        adamw_lr_ratio=adamw_lr / lr, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, adamw_wd=adamw_wd)

        params = list(kronos_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        for p in kronos_params:
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]['use_kronos'] = True
            else:
                self.state[p]['use_kronos'] = False
        for p in adamw_params:
            self.state[p]['use_kronos'] = False

        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
        else:
            self.world_size = 1
            self.rank = 0

        # Dictionary to store Q matrices keyed by column dimension
        self.shared_Qs = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Kronos processing
            params = [p for p in group['params'] if self.state[p]['use_kronos']]
            lr = group['lr']
            momentum = group['momentum']
            whitening_prob = group['whitening_prob']

            for p in params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                m, n = g.shape
                if m < n:
                    g = g.T
                
                # Get dimension-based Q matrix
                n_cols = g.shape[1]
                device = g.device
                
                if n_cols not in self.shared_Qs:
                    # Initialize new Q for this column dimension
                    self.shared_Qs[n_cols] = torch.eye(n_cols, device=device)
                
                Q = self.shared_Qs[n_cols]

                if torch.rand(1).item() < whitening_prob:
                    Q = single_sided_whitening(torch.sign(g), Q, lr_param=group['lr_param'])
                    self.shared_Qs[n_cols] = Q

                # Apply transformation
                g = torch.sign(g) @ Q.T @ Q
                if m < n:
                    g = g.T
                g *= max(1, g.size(0) / g.size(1)) ** 0.5

                p.data.add_(g, alpha=-lr)

            # AdamW processing
            params = [p for p in group['params'] if not self.state[p]['use_kronos']]
            lr = group['adamw_lr_ratio'] * group['lr']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            weight_decay = group['adamw_wd']

            for p in params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['moment1'] = torch.zeros_like(g)
                    state['moment2'] = torch.zeros_like(g)
                state['step'] += 1

                buf1 = state['moment1']
                buf2 = state['moment2']
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                scale = bias_correction1 / bias_correction2 ** 0.5

                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
