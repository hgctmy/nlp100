import os
import torch
from torch import nn
from torch.utils.data import dataloader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

input_image = torch.rand(3, 28, 28)
print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

'''
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
    (5): ReLU()
  )
)
Predicted class: tensor([5])
torch.Size([3, 28, 28])
torch.Size([3, 784])
torch.Size([3, 20])
Before ReLU: tensor([[-4.4892e-01, -2.9037e-02, -2.5245e-01, -3.6707e-01, -3.3908e-01,
          6.5474e-01,  3.5450e-02, -2.2190e-01,  1.2138e-01,  1.0376e-02,
         -2.3496e-01, -7.6709e-02, -6.0672e-01, -9.0188e-02,  2.5740e-01,
          5.5192e-02, -4.3236e-01, -5.6209e-01,  4.5341e-01,  6.0384e-01],
        [-1.0930e-01, -1.5555e-01,  6.0527e-03,  9.1253e-02, -5.8391e-01,
          4.3305e-01,  1.0940e-01, -3.2423e-01, -2.8138e-02, -3.7904e-02,
         -4.0681e-01, -2.0037e-01, -5.1372e-01, -1.1323e-01,  2.3786e-02,
         -8.1980e-02, -6.0359e-02, -4.2439e-01,  2.2006e-01,  1.1877e-01],
        [ 5.7669e-02,  2.7578e-02, -2.5424e-01, -3.5502e-01, -6.1148e-01,
          3.2129e-01, -1.3532e-01, -1.3552e-01,  2.5061e-01, -1.3386e-01,
         -2.1090e-01, -2.4673e-01, -3.1979e-01, -9.7770e-02, -1.3853e-01,
          1.7801e-01, -3.8009e-01, -6.5498e-01,  3.0100e-01, -4.8084e-04]],
       grad_fn=<AddmmBackward>)


After ReLU: tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6547, 0.0354, 0.0000, 0.1214,
         0.0104, 0.0000, 0.0000, 0.0000, 0.0000, 0.2574, 0.0552, 0.0000, 0.0000,
         0.4534, 0.6038],
        [0.0000, 0.0000, 0.0061, 0.0913, 0.0000, 0.4331, 0.1094, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0238, 0.0000, 0.0000, 0.0000,
         0.2201, 0.1188],
        [0.0577, 0.0276, 0.0000, 0.0000, 0.0000, 0.3213, 0.0000, 0.0000, 0.2506,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1780, 0.0000, 0.0000,
         0.3010, 0.0000]], grad_fn=<ReluBackward0>)
Model structure:  NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
    (5): ReLU()
  )
) 


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[-0.0180, -0.0239, -0.0194,  ...,  0.0178,  0.0346,  0.0006],
        [-0.0293, -0.0346,  0.0316,  ...,  0.0104,  0.0227, -0.0357]],
       grad_fn=<SliceBackward>) 

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0166,  0.0279], grad_fn=<SliceBackward>) 

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0210, -0.0411,  0.0147,  ...,  0.0334, -0.0358,  0.0219],
        [ 0.0345,  0.0329, -0.0117,  ..., -0.0161, -0.0310,  0.0068]],
       grad_fn=<SliceBackward>) 

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([-0.0258, -0.0377], grad_fn=<SliceBackward>) 

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0125, -0.0403,  0.0055,  ..., -0.0057,  0.0121,  0.0055],
        [ 0.0062,  0.0413, -0.0372,  ..., -0.0207, -0.0196,  0.0069]],
       grad_fn=<SliceBackward>) 

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([-0.0371,  0.0040], grad_fn=<SliceBackward>) 

'''
