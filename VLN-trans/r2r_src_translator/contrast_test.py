# constrastive_demo.py
# "Dimensionality Reduction by Learning an Invariant Mapping"

# PyTorch 1.10.0-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10 /11

import numpy as np
import torch as T
device = T.device('cpu')


# -----------------------------------------------------------

class ContrastiveLoss(T.nn.Module):
  def __init__(self, m=2.0):
    super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
    self.m = m  # margin or radius

  def forward(self, y1, y2, d=0):
    # d = 0 means y1 and y2 are supposed to be same
    # d = 1 means y1 and y2 are supposed to be different
    
    euc_dist = T.nn.functional.pairwise_distance(y1, y2)

    if d == 0:
      return T.mean(T.pow(euc_dist, 2))  # distance squared
    else:  # d == 1
      delta = self.m - euc_dist  # sort of reverse distance
      delta = T.clamp(delta, min=0.0, max=None)
      return T.mean(T.pow(delta, 2))  # mean over all rows
    
# -----------------------------------------------------------

def main():
  print("\nBegin contrastive loss demo \n")

  loss_func = ContrastiveLoss()

  y1 = T.tensor([[1.0, 2.0, 3.0, 0.0, 0.0],
                 [3.0, 4.0, 5.0, 0.0, 0.0]], dtype=T.float32).to(device)

  y2 = T.tensor([[1.0, 2.0, 3.0, 0.0, 0.0],
                 [3.0, 4.0, 5.0, 0.0, 0.0]], dtype=T.float32).to(device)

  y3 = T.tensor([[10.0, 20.0, 30.0, 0.0, 0.0],
                 [30.0, 40.0, 50.0, 0.0, 0.0]], dtype=T.float32).to(device)

  loss = loss_func(y1, y2, 0)
  print(loss)  # 0.0 -- small; y1 y2 should be equal

  loss = loss_func(y1, y2, 1)   
  print(loss)  # 4.0 -- large; y1 y2 should be different

  loss = loss_func(y1, y3, 0)  
  print(loss)  # 2591.99 -- large; y1 y3 should be equal

  loss = loss_func(y1, y3, 1)  
  print(loss)  # 0.0 -- small; y1 y2 should be different 

  print("\nEnd demo ")

if __name__ == "__main__":
  main()