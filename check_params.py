from models.simple_cnn_mnist import Head, Backbone, Tail
import torch.nn as nn

def count_params(model):
    return sum(p.numel() for p in model.parameters())

head, backbone, tail = Head(), Backbone(), Tail()
print("Head params:", count_params(head))
print("Backbone params:", count_params(backbone))
print("Tail params:", count_params(tail))
print("Total params:", count_params(head)+count_params(backbone)+count_params(tail))