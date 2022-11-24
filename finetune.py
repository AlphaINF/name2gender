from utils import *
from name2gender import *
from test import *
import torch.optim as optim

data = csv_loader('dataset/train.csv')
data_set = NameDataset(data)
batch_size = 64
data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

model = name2gender(32768, 256, 128, 50)
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
loss_func = nn.NLLLoss()

total_loss = 0
total_step = 0

all_run_times = 0

best_acc = 0

while True:
    for data in data_loader:
        total_step += 1
        names = data[0]
        labels = data[1]

        name_tensors, labels, name_lengths = value_to_tensor(names, labels)
        out = model(name_tensors, name_lengths).view(-1,2)
        loss = loss_func(out, labels)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if total_step % 50 ==0:
            print('%dth step, avg_loss: %0.4f'%(total_step*batch_size, total_loss/batch_size))
            total_loss = 0
            out = torch.argmax(out, dim=-1)
            result = (out == labels).to(torch.float)
            print(torch.mean(result))

        if total_step %500==0:
            runtimes = total_step * batch_size
            print('modelsave',runtimes)
            torch.save(model, str(runtimes)+'net.pth')
            acc = test_acc(model)
            if acc > best_acc :
                best_acc = acc
                torch.save(model,'best_net.pth')
                print('Updating best model')
            #break

    all_run_times += 1
    torch.save(model, str(all_run_times) + 'net.pth')