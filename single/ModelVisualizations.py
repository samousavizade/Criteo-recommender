import torchviz
import mlflow as mlf
import torch

experiment_id = 3
run_id = '4f95dd52487d4f959b980050d1ea0f98'
# path = f'./mlruns/{experiment_id}/{run_id}'
path = f'./mlruns/3/4f95dd52487d4f959b980050d1ea0f98/artifacts/Extreme Deep Factorization Model/'
model = mlf.pytorch.load_model(path)

x = torch.randn(1, 14)
y = model(x)

torchviz.make_dot(y.mean(), params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
