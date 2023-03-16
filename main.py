import csv
from utils import *

model = torch.load('net.pth')
model.eval()

def test(name):
    name_tensor, len_tensor = name_to_tensor(name)
    out = model(name_tensor, len_tensor)
    ans = int(torch.argmax(out, dim=-1))
    return ['female', 'male'][ans]

# 读取姓名的函数
def read_names_from_csv(input_file):
    with open(input_file, "r", encoding='ansi') as f:
        reader = csv.reader(f)
        names = [row[0] for row in reader]
    return names

# 将姓名和预测的性别写入 CSV 的函数
def write_predictions_to_csv(output_file, names, predictions):
    with open(output_file, "w", encoding='ansi', newline='') as f:
        writer = csv.writer(f)
        for name, prediction in zip(names, predictions):
            writer.writerow([name, prediction])

# 读取输入文件中的姓名
input_file = "in.csv"
names = read_names_from_csv(input_file)

# 对读取的姓名进行性别预测
predictions = [test(name) for name in names]

# 将姓名和预测的性别写入输出文件
output_file = "out.csv"
write_predictions_to_csv(output_file, names, predictions)