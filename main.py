from utils import *

model = torch.load('net.pth')
model.eval()

def test(name):
    name_tensor, len_tensor = name_to_tensor(name)
    out = model(name_tensor, len_tensor)
    out = torch.exp(out)
    ans = int(torch.argmax(out, dim=-1))
    print('({}, {})'.format(['female', 'male'][ans], float(out[0][ans])))

while True:
    name = input('input your name:\n')
    test(name)