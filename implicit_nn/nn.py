import torch
import torch.nn as nn

def broyden_method(f, x0, tol=1e-6, max_iter=100):
    # f: 함수, 비선형 시스템의 목표 f(x) = 0
    # x0: 초기 추정값
    # tol: 허용 오차
    # max_iter: 최대 반복 횟수

    x = x0                            # 초기 추정값
    J_inv = approx_inverse_jacobian(f, x)  # 초기 역 Jacobian 근사값
    
    for k in range(max_iter):
        Fx = f(x)                     # 현재 추정값에서의 함수 값
        if norm(Fx) < tol:
            return x                  # 수렴한 경우, 추정값 반환

        x_new = x - (J_inv @ Fx)                # 새로운 추정값 계산

        # Broyden 업데이트 (rank-1 update to J_inv)
        delta_x = x_new - x           # x 변화량
        delta_f = f(x_new) - Fx       # f(x) 변화량
        J_inv = J_inv + ((delta_x - J_inv @ delta_f) @ delta_x.T) / (delta_x.T @ delta_x)

        x = x_new                     # 다음 반복을 위해 새로운 추정값 업데이트

    raise ValueError("Broyden's method did not converge within the maximum number of iterations.")


# @torch.jit.script
def compute_broyden_update(
        B: torch.Tensor,
        deltaZ: torch.Tensor,
        deltaG: torch.Tensor
    ) -> torch.Tensor:
    """
    This is eq 10 in the paper
    :param B: nxn inv jacobian approx
    :param deltaZ: n vector
    :param deltaG: n vector
    """
    Bdg = torch.mv(B, deltaG)  # n
    rational = (deltaZ - Bdg).div_(torch.dot(deltaZ, Bdg).clamp_(1e-10))
    notrational = torch.matmul(deltaZ, B)  # n
    update = torch.outer(rational, notrational)
    return update


class FixedpointLayer(torch.autograd.Function):
    """
    given a function G and an input x, return z_star such that 
    z_star = G(z_star, x). This is the fixed point of G.
    """

    @staticmethod
    def broyden_solver(G, z_0, eps, alpha, max_iters, verbose=True):
        """
        This function performes broyden's method, finding the root of g.
        This function is used by both the forward and backward pass.
        :param z_0: the first estimate of a root
        :param eps: the error tolerance of an accepted root.
        :param alpha: the step size
        :param max_iters: the maximum number of broyden steps
        """
        z = z_0
        B = torch.eye(z.size(0))  # jacobian_inverse matrix

        for _ in range(max_iters):
            g = G(z)
            z_update = - alpha * torch.matmul(B, g)
            z.add_(z_update)

            # update B matrix
            B_update = compute_broyden_update(B, z_update, G(z) - g)
            B.add_(B_update)

            if torch.abs(z_update).max() < eps:
                break
        else:
            if verbose:
                print("Broyden's method did not converge.")
        return z

    @staticmethod
    def lbfgs_solver(G, z_0, eps, alpha, max_iters, verbose=True):
        z = z_0
        z.requires_grad_(True)
        optimizer = torch.optim.LBFGS([z], lr=alpha, history_size=5)

        def closure():
            optimizer.zero_grad()
            loss = (z - G(z)).norm()
            loss.backward()
            return loss

        for _ in range(max_iters):
            optimizer.step(closure)

            with torch.no_grad():
                if torch.allclose(z, G(z), atol=eps):
                    break
        else:
            if verbose:
                print("L-BFGS did not converge.")
        return z


    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, G, x, eps, alpha, max_iters):
        """
        :param func:
        :returns: equilibrium z_star, the root of f(z,x)
        """
        z_0 = torch.zeros_like(x)
        root = FixedpointLayer.broyden_solver(G, z_0, eps, alpha, max_iters)

        """
        :param g: the pytorch function that we've found the root to.
        :param z_star: the root of g, found arbitrarily.
        """

        ctx.G = G
        ctx.save_for_backward(root)

        return root

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dl_dzstar):
        """
        Input into this backward is the gradient of the loss wrt the equilibrium.
        From here, we want to pass a gradient to f, which in turn will pass it
        to the parameters within f. We can create this gradient however we want;
        we don't need torch ops, because we are the torch op.
        """
        z_star, = ctx.saved_tensors
        G = ctx.G
        z_shape = z_star.size()

        with torch.enable_grad():
            # copy z_star, detach the copy from any graph,
            # and then enforce grad functionality
            z_star = z_star.reshape((-1,)).clone().detach().requires_grad_()
            # y and z_star, at this point, are both 1D vectors
            y = G(z_star)

        dl_dzstar_flat = dl_dzstar.reshape((-1,))
        # this function represents the LHS of eq 11 in the original paper
        # we use autograd to calculate the Jacobian-vector product
        def JacobianVector(x):
            y.backward(x, retain_graph=True)
            JTxT = z_star.grad.clone().detach()
            z_star.grad.zero_()  # remove the gradient (this is kind of a fake grad)
            return JTxT + dl_dzstar_flat

        neg_dl_dzs_J_inv = FixedpointLayer.broyden_solver(
            JacobianVector,
            torch.zeros_like(z_star),
            2e-7,
            alpha=0.5,
            max_iters=200
        )


        return None, neg_dl_dzs_J_inv.reshape(z_shape)


class DEQResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, 32)
        self.groupnorm2 = nn.GroupNorm(32, 32)
        self.groupnorm3 = nn.GroupNorm(32, 32)
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.act = nn.ReLU()
    
    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z0 = z
        z = self.groupnorm1(z)
        z = self.conv1(z)
        z = self.act(z)
        z = self.conv2(z)
        z = z + x
        z = self.groupnorm2(z)
        z = z + z0
        z = self.act(z)
        z = self.groupnorm3(z)
        return z


class DEQ(nn.Module):
    """
    A module to represent a DEQ.
    """
    def __init__(self, f, eq_shape, forward_eps, backward_eps,
                 alpha, max_iters, root_find_method):
        """
        :param forward_eps: epsilon for the equilibrium in the forward pass
        :param backward_eps: epsilon for the equilibrium in the backward pass
        :param f: The Pytorch differentiable function that will be "iterated".
        This could be a nn.Module, or a Functional layer. Should have parameters
        f takes two parameters: z, the equilibrium, and x the model input.
        The output of f must be the same shape as z.
        f can take any shape of input, but broyden's method will be applied
        per-batch. That is, broyden's method only terminates once all examples
        in the batch have converged.
        :param alpha: step size for broyden's method
        """
        super(DEQ, self).__init__()
        self.f = f
        self.forward_eps = forward_eps
        self.RootFind = root_find_method
        self.alpha = alpha
        self.max_iters = max_iters
        self.eq_shape = list(eq_shape)

    def forward(self, x):
        """
        :param x: Tensor with unspecified size, however the first dimension is
        a batch dimension
        """
        batch_size = x.size()[0]
        self.eq_shape[0] = batch_size

        def g(z):
            """
            This is the function we will find the root of. The root is equal
            to the equilibrium of f.
            :param z_i: a vector in the equilibrium space (1D)
            :returns z_{i+1}: a vector in the equilibrium space (1D)
            """
            if len(z.size()) != 1:
                raise ValueError(f"g only takes 1D tensors, but a tensor of \
                                 {z.size()} was provided.")
            z_reshaped = torch.reshape(z, self.eq_shape)
            result = self.f(z_reshaped, x) - z_reshaped
            result_flat = torch.reshape(result, (-1,))
            return result_flat

        with torch.no_grad():
            z_0 = torch.zeros(self.eq_shape).reshape((-1,))
            z_star_flat = self.RootFind.apply(g, z_0, self.forward_eps,
                                         self.alpha, self.max_iters)

        # this is a call to pass the gradient to the parameters of f.
        z_star = torch.reshape(z_star_flat, self.eq_shape)
        z_star = self.f(z_star, x)
        # this call doesn't modify z_star, but ensures we differentiate
        # properly in the backward pass.
        z_star = FixedpointLayer.apply(g, z_star)

        return z_star


def finite_diff_grad_check():
    # device = torch.device("cuda")
    device = torch.device("cpu")

    get_mlp = lambda: nn.Sequential(
        nn.Linear(4, 2, bias=False),
        nn.Tanh(),
    )
    input = torch.rand(1, 4, requires_grad=True, dtype=torch.float64, device=device)
    breakpoint()
    output = FixedpointLayer.apply(get_mlp().double(), input, 1e-5, 0.1, 10)

    model = DEQ(
        f=get_mlp(),
        eq_shape=(2,),
        forward_eps=1e-6,
        backward_eps=1e-6,
        alpha=0.5,
        max_iters=100,
        root_find_method=FixedpointLayer
    )

    model.to(dtype=torch.float64, device=device)
    
    # @torch.autocast("cuda", dtype=torch.bfloat16)
    # @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def forward_loss_fn(x):
        x = model(x)
        loss = x.norm()
        return loss

    input = torch.rand(1, 4, requires_grad=True, dtype=torch.float64, device=device)

    if torch.autograd.gradcheck(forward_loss_fn, input, nondet_tol=1e-5):
        print("Gradient check passed!")


if __name__ == "__main__":
    finite_diff_grad_check()
