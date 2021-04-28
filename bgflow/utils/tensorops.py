def log_dot_exp(logA, logB):
    """Fast and stable matrix log multiplication"""
    maxA = logA.max(dim=-1, keepdim=True).values
    maxB = logB.max(dim=-2, keepdim=True).values
    A = (logA - maxA).exp()
    B = (logB - maxB).exp()
    batch_shape = A.shape[:-2]
    A = A.view(-1, A.shape[-2], A.shape[-1])
    B = B.view(-1, B.shape[-2], B.shape[-1])
    logC = A.bmm(B).log().view(*batch_shape, A.shape[-2], B.shape[-1])
    logC.add_(maxA + maxB)
    return logC
