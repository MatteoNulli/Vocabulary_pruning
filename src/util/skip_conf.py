import torch
from transformers import AutoConfig


def softmax_confidence(
    logits: torch.Tensor,
):
    probs = torch.softmax(logits, dim=-1)
    top_2 = torch.topk(probs, dim=-1, k=2)[0]

    return (top_2[..., 0] - top_2[..., 1]).squeeze()

def get_skip_mask(
    logits: torch.Tensor = None,
    config: AutoConfig = None,
    return_conf=False,
):
    """
    Get the skip mask based on the confidence measure and threshold.
    """

    if config.exit_conf_type is not None:
        threshold = config.exit_conf_threshold

    #conf_measure = get_confidence_class(key=key)
    # since its only softmax, we can directly use the function
    conf_measure = softmax_confidence
    conf = conf_measure(logits=logits)
    mask = torch.where(conf <= threshold, 0.0, 1.0).bool()
    if not return_conf:
        return mask.item()  # False (0) and True (1) denote keep and exit
    else:
        return mask.item(), conf.item()
