from utils import *

def test_acc(model):
    #print('Start evluating acc')

    data = ccnc_loader('dev.csv')
    data_set = NameDataset(data)
    batch_size = 128
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

    sum_acc = 0
    sum = 0
    step = 0

    for data in data_loader:
        names = data[0]
        labels = data[1]
        step += 1

        name_tensors, labels, name_lengths = value_to_tensor(names, labels)
        out = model(name_tensors, name_lengths).view(-1, 2)

        out = torch.argmax(out, dim=-1)
        result = (out == labels).to(torch.float)
        sum_acc += torch.sum(result)
        sum += len(out)

        #if step%50 ==0 :
        #    print("Case {} : acc={}".format(step*batch_size,sum_acc/sum))

    acc = sum_acc/sum
    #print('acc=',acc)
    return acc