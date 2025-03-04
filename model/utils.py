import torch
import torch.nn.functional as F


def get_model_fn(model, train=False):
    input_dim = (model.module if hasattr(model, "module") else model).input_dim
    def model_fn(x, t):
        if train:
            model.train()
        else:
            model.eval()
        return model(x[...,:input_dim], t)
    return model_fn


def get_drift_fn(model, sde, train=False, sampling=False, **kwargs):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        def drift_fn(x, t):
            probs = F.softmax(model_fn(x, t).to(torch.float32), dim=-1)
            probs = torch.cat([probs, torch.zeros((*probs.shape[:-1], x.shape[-1]-probs.shape[-1]), device=x.device)], dim=-1)
            
            drift = sde.manifold.weighted_sum(probs, x)
            drift = sde.scale_by_coeff(drift, t)
            drift = sde.manifold.to_tangent(drift, x)
            return drift

    return drift_fn