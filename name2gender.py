import torch.nn as nn

class name2gender(nn.Module):
    def __init__(self, input_size, embedding_size, rnn_hidden_size, hidden_size, output_size=2):
        super(name2gender, self).__init__()
        self.embeddings = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.drop = nn.Dropout(p=0.1)
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=rnn_hidden_size, batch_first=True)
        self.linear1 = nn.Linear(rnn_hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.output_size = output_size
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, name, length):
        now = self.embeddings(name)
        now = self.drop(now)
        _, (ht, _) = self.rnn(now, None)
        out = self.linear1(ht)
        out = self.activation(out)
        out = self.linear2(out)

        out = out.view(-1, self.output_size)
        out = self.softmax(out)
        return out

