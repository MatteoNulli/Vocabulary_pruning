import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoConfig
from copy import deepcopy


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.Tensor, q: torch.Tensor):
        # Move p and q to CPU and ensure they are in float64 for high precision calculation
        p, q = p.cpu().double(), q.cpu().double()
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
    
def softmax_confidence(
    logits: torch.Tensor = None,
):  
    # start = datetime.datetime.now()
    assert logits is not None
    probs = torch.softmax(logits, dim=-1)
    top_2 = torch.topk(probs, dim=-1, k=2)[0]
    # end = datetime.datetime.now()
    # print("Time taken for softmax confidence", end-start)

    return (top_2[..., 0] - top_2[..., 1]).squeeze()
    

def meta_confidence(
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
):
    assert hidden_states is not None
    assert classifier is not None
    
    preds = classifier(hidden_states)
    probs = torch.softmax(preds, dim=-1)
    return probs[..., 1].squeeze()


def contrastive_confidence(  
    lm_logits: torch.Tensor = None,
    layer_exp: int = None, 
    prev_probits: dict = None, 
    layer_am: int = None,
    alpha: float = None,
):
    """
    Checking confidence with contrastive decoding.
    First we are computing the V_head, meaning the plausibility constraint.
    Second we are getting the probits from previous iterations and comparing with a fixed one by taking the log difference.
    
    """

    assert lm_logits is not None


    ## calculate current layer probabilities
    # start = datetime.datetime.now()
    probits_exp = torch.softmax(lm_logits, dim=-1)
    probits_exp = torch.squeeze(probits_exp)
    prev_probits[layer_exp] = probits_exp
   
    # probs_exp = torch.softmax(logits_at, dim=-1)
    max_probs_exp = torch.max(probits_exp)

    ## obtaining the correct layer probit values from previous layers (the layer index is choosen to be usually half of the current layer). 
    if layer_am in prev_probits.keys():
        probits_am = prev_probits[layer_am]
    else:
        raise ValueError("Choosen layer has not been computed yet")
    
    ## calculating the scores using the plausibility constraint
    s = deepcopy(probits_exp)

    mask = probits_exp >= alpha * max_probs_exp
    s[mask] = torch.softmax(torch.log(probits_exp[mask]) - torch.log(probits_am[mask]), dim=-1) 
    top_2 = torch.topk(s, dim=-1, k=2)[0]
    # end = datetime.datetime.now()
    # print("Time taken for contrastive confidence", end-start)
    
    return (top_2[..., 0] - top_2[..., 1]).squeeze()

def reweight_contrastive_confidence(  
    lm_logits: torch.Tensor = None,
    layer_exp: int = None, 
    prev_probits: dict = None, 
    layer_am: int = None,
    alpha: float = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
):
    """
    Checking confidence with reweighted contrastive decoding.
    First we are computing the V_head, meaning the plausibility constraint.
    Second we are getting the probits from previous iterations and comparing with a fixed one by taking the log difference.
    
    """
    # start = datetime.datetime.now()
    assert lm_logits is not None


    ## calculate current layer probabilities
    probits_exp = torch.softmax(lm_logits, dim=-1).squeeze_()
    prev_probits[layer_exp] = probits_exp
    # print("prev_probits", prev_probits) 
    # probs_exp = torch.softmax(logits_at, dim=-1)
    max_probs_exp = torch.max(probits_exp)

    ## obtaining the correct layer probit values from previous layers (the layer index is choosen to be usually half of the current layer). 
    if layer_am in prev_probits.keys():
        probits_am = prev_probits[layer_am]
    else:
        raise ValueError("Choosen layer has not been computed yet")


    s = torch.zeros_like(probits_exp)
    mask = probits_exp >= alpha * max_probs_exp

    # start = datetime.datetime.now()
    contrast = torch.softmax(torch.log(probits_exp[mask]) - torch.log(probits_am[mask]), dim=-1).mul_(torch.sum(probits_exp[mask]))
    s[mask] = contrast
    
    top_2 = torch.topk(s, dim=-1, k=2)[0]
    # end = datetime.datetime.now()
    # print("Time taken for contrastive confidence", end-start)
    
    return (top_2[..., 0] - top_2[..., 1]).squeeze()
 
def JSD_contrastive_confidence(  
    lm_logits: torch.Tensor = None,
    layer_exp: int = None, 
    prev_probits: dict = None, 
    alpha: float = None,
    return_jsds=False,
):
    """
    Checking confidence with JSD contrastive decoding.
    First we are computing the V_head, meaning the plausibility constraint.
    Second we are getting the probits from previous iterations and comparing with a fixed one by taking the log difference.
    
    """
    # start = datetime.datetime.now()
    assert lm_logits is not None
    ## calculate current layer probabilities
    probits_exp = torch.softmax(lm_logits, dim=-1).squeeze_()
    prev_probits[layer_exp] = probits_exp
    # print("prev_probits", prev_probits) 
    max_probs_exp = torch.max(probits_exp)
    jsd = JSD()
    # only consider jsds between current and 2nd layer
    mask = probits_exp >= alpha * max_probs_exp
    jsds = {layer: jsd(probits_exp[mask], prev_probits[layer][mask]) / (layer_exp - layer) for layer in np.arange(stop = layer_exp, start=2)}

    # get the probits with the maximum jsd
    max_jsd_layer = max(jsds, key=jsds.get)
    probits_am = prev_probits[max_jsd_layer]
    s = torch.zeros_like(probits_exp)
    contrast = (torch.log(probits_exp[mask]) - torch.log(probits_am[mask])) #/ 2.5 # temperature scaling
    s[mask] = contrast
    s[mask] = torch.softmax(s[mask], dim=-1).mul_(torch.sum(probits_exp))
    s[~mask] = probits_exp[~mask]
    top_2 = torch.topk(s, dim=-1, k=2)[0]
    
    if return_jsds:
        return (top_2[..., 0] - top_2[..., 1]).squeeze(), jsds
    else:
        return (top_2[..., 0] - top_2[..., 1]).squeeze()


def get_confidence_class(key):

    _conf_class_map = {
        'softmax': softmax_confidence,
        'meta': meta_confidence,
        'contrastive_decoding': contrastive_confidence,
        'reweight_contrastive_decoding': reweight_contrastive_confidence,
        'JSD_contrastive_confidence':  JSD_contrastive_confidence,
    }

    if key in _conf_class_map:
        return _conf_class_map[key]
    else:
        raise ValueError('Invalid confidence measure: {}'.format(key))

def get_skip_mask_cd(
    lm_logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    config: AutoConfig = None,
    pos_time: int = 1,
    layer_exp: int = None, 
    prev_probits: dict = None, 
    layer_am: int = None,
    alpha: float = 0.1,
    adapt_threshold: float = None,
    return_conf=False,
    return_jsds=False,
):

    assert config.exit_conf_type is not None or config.shallow2deep_conf_type is not None

    if config.exit_conf_type is not None:
        key = config.exit_conf_type
        if config.exit_position_temp is not None:
            # decays the confidence threshold with decoding time stp.        
            correct_by_pos = lambda i: config.exit_conf_threshold * np.exp(
                - config.exit_position_temp * i / config.max_answer_length
            ) / 10 + 9 * config.exit_conf_threshold / 10
            threshold = correct_by_pos(pos_time)
        else:
            threshold = config.exit_conf_threshold
    elif config.shallow2deep_conf_type is not None:
        key = config.shallow2deep_conf_type
        threshold = config.shallow2deep_conf_threshold if adapt_threshold is None else adapt_threshold

    conf_measure = get_confidence_class(key=key)    

    # print("Inside get_skip_mask_cd")

    if key == 'JSD_contrastive_confidence' and not return_jsds:
        conf = conf_measure(
            lm_logits,
            layer_exp = layer_exp, 
            prev_probits = prev_probits, 
            alpha = alpha,
        )
    elif key == 'JSD_contrastive_confidence' and return_jsds:
        conf, jsds = conf_measure(
            lm_logits,
            layer_exp = layer_exp, 
            prev_probits = prev_probits, 
            alpha = alpha,
            return_jsds = return_jsds,
        )
    else:
        conf = conf_measure(
            lm_logits,
            layer_exp = layer_exp, 
            prev_probits = prev_probits, 
            layer_am = layer_am,
            alpha = alpha,
            hidden_states = hidden_states,
            classifier = classifier,
        )
    


    # print("confidence return", conf)

    mask = torch.where(conf <= threshold, 0., 1.).bool()

    # print("Are we early exiting?", mask.item() == 1)
    # print('Confidence:', conf.item(), 'Threshold:', threshold, 'Mask:', mask.item())

    # print("mask", mask)
    # print("mask shape", mask.shape)

    if return_jsds and return_conf:
        return mask.item(), jsds, conf.item()
    elif return_jsds and not return_conf:
        return mask.item(), jsds  # False (0) and True (1) denote keep and exit
    elif not return_jsds and return_conf:
        return mask.item(), conf.item()
    else:
        return mask.item()
    


def get_skip_mask(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    config: AutoConfig = None,
    pos_time: int = 1,
    adapt_threshold: float = None,
    return_conf=False,
):
    assert config.exit_conf_type is not None or config.shallow2deep_conf_type is not None

    if config.exit_conf_type is not None:
        key = config.exit_conf_type
        if config.exit_position_temp is not None:
            # decays the confidence threshold with decoding time stp.        
            correct_by_pos = lambda i: config.exit_conf_threshold * np.exp(
                - config.exit_position_temp * i / config.max_answer_length
            ) / 10 + 9 * config.exit_conf_threshold / 10
            threshold = correct_by_pos(pos_time)
        else:
            threshold = config.exit_conf_threshold
    elif config.shallow2deep_conf_type is not None:
        key = config.shallow2deep_conf_type
        threshold = config.shallow2deep_conf_threshold if adapt_threshold is None else adapt_threshold

    conf_measure = get_confidence_class(key=key)    
    
    conf = conf_measure(
        logits=logits
        )
    
    mask = torch.where(conf <= threshold, 0., 1.).bool()

    # print(f"Confidence: {conf.item():.4f}, Threshold: {threshold:.4f}, Mask: {mask.item()}")
    
    # print("Are we early exiting?", mask.item() == 1)
    #print('Confidence:', conf.item(), 'Threshold:', threshold, 'Mask:', mask.item())


    if not return_conf:
        return mask.item()  # False (0) and True (1) denote keep and exit
    else:
        return mask.item(), conf.item()