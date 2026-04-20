import minitorch as mt

x = mt.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = mt.tensor([[0.5, -1.0], [2.0, 1.5]], requires_grad=True)

mt.grad_check(lambda a, b: a * b + a, x, y)
print("grad_check passed")