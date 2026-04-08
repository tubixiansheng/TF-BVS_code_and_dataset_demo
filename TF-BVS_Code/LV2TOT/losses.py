def pair_align_loss(hat_T, E_T, w_cos=1.0, w_l2=0.0):
    loss = 0.0
    if w_cos > 0:
        loss += w_cos * (1 - F.cosine_similarity(hat_T, E_T, dim=-1)).mean()
    if w_l2 > 0:
        loss += w_l2 * F.mse_loss(hat_T, E_T)
    return loss

def ot_cost_matrix_label_only(hat_T, E_T, labels, lambda_g=1.0):  
    B, d = hat_T.shape
    dist_feat = torch.cdist(hat_T, E_T, p=2).pow(2)  # [B, B]
    label_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()   # [B, B]
    penalty = lambda_g * label_mask
    cost_matrix = dist_feat + penalty
    return cost_matrix  # [B, B]


def mmd_rbf(X, Y, sigma=1.0):
    # X: [Bx, d], Y: [By, d]
    XX = torch.cdist(X, X, p=2).pow(2)
    YY = torch.cdist(Y, Y, p=2).pow(2)
    XY = torch.cdist(X, Y, p=2).pow(2)
    Kxx = torch.exp(-XX/(2*sigma*sigma))
    Kyy = torch.exp(-YY/(2*sigma*sigma))
    Kxy = torch.exp(-XY/(2*sigma*sigma))

    mmd = Kxx.mean() + Kyy.mean() - 2*Kxy.mean()
    return mmd

def temporal_tv(T_seq, flow_mask=None):
    # T_seq: [B, T, d] or [B, d] if no time dimension
    if T_seq.dim() == 2:  # If only [B, d], no temporal TV
        return T_seq.new_zeros(())

    diff = T_seq[:, 1:, :] - T_seq[:, :-1, :]
    if flow_mask is not None:
        # flow_mask: [B, T-1, 1], higher values reduce regularization
        diff = diff * (1.0 - flow_mask)
    return diff.abs().mean()
