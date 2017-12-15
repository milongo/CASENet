import torch.nn as nn
import torch


class MultiLLFunction(nn.Module):
    def __init__(self, beta=0.5):
        super(MultiLLFunction, self).__init__()
        self.beta = beta

    def forward(self, predictions, targets):
        """Multilabel loss
                Args:
                    predictions: a tensor containing pixel wise predictions
                        shape: [batch_size, num_classes, width, height]
                    targets: a tensor containing binary labels
                        shape: [batch_size, num_classes, width, height]
                """
        log1 = torch.log(predictions)
        log2 = torch.log(1 - predictions)
        term1 = torch.mul(torch.mul(targets, -self.beta), log1)
        term2 = torch.mul(torch.mul(1-targets, 1-self.beta), log2)
        sum_of_terms = term1 - term2
        return torch.sum(sum_of_terms)
