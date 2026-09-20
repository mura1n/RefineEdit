import torch

def index_label2quant_features(pred_sample_labels, hbq_round):
    approx_signal = 0.
    pred_sample_labels = pred_sample_labels.to(torch.long)
    for round_ind in range(hbq_round):
        interval = (1/2)**(round_ind+1)
        base = 2**(hbq_round-1-round_ind)
        approx_signal = approx_signal + interval * torch.where(pred_sample_labels>=base, +1, -1)
        pred_sample_labels = pred_sample_labels % base
    return approx_signal

def raw_feature2index_label(feature, hbq_round):
    quant_features = 0.
    labels = 0
    for round_ind in range(hbq_round):
        interval = (1/2) ** (round_ind + 1)
        labels = labels * 2 + torch.where(feature > quant_features, 1, 0)
        quant_features = quant_features + torch.where(feature > quant_features, interval, -interval)
    return labels

def raw_feature2bit_label(feature, hbq_round):

    quant_features = 0.
    labels = []
    for round_ind in range(hbq_round):
        interval = (1/2) ** (round_ind + 1)
        labels.append(torch.where(feature > quant_features, 1, 0))
        quant_features = quant_features + torch.where(feature > quant_features, interval, -interval)
    labels = torch.stack(labels, dim=1)
    B, _, d, t, h, w = labels.shape
    labels = labels.reshape(B, hbq_round * d, t, h, w)
    return labels

def bit_label2raw_feature(bit_labels, hbq_round):

    B, hbq_round_mul_d, t, h, w = bit_labels.shape
    d = hbq_round_mul_d // hbq_round
    bit_labels = bit_labels.reshape(B, hbq_round, d, t, h, w).to(torch.long)
    raw_features = 0.
    for round_ind in range(hbq_round):
        interval = (1/2) ** (round_ind + 1)
        raw_features = raw_features + interval * torch.where(bit_labels[:,round_ind] == 1, +1, -1)
    return raw_features

def multiclass_labels2onehot_input(labels, num_classes):
    B,d,t,h,w = labels.shape
    onehot_input = torch.nn.functional.one_hot(labels.to(torch.long), num_classes)
    onehot_input = onehot_input.permute(0,1,5,2,3,4).reshape(B,d*num_classes,t,h,w).float()
    return onehot_input
