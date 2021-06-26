import os
from torch.utils.tensorboard import SummaryWriter
import time
import math
import torch
import torch.nn as nn
import torch.onnx

import dataset
from model import Model

sequence_length = 32
clip = 0.25
log_interval = 50
input_size = 512
hidden_size = 512
num_layers = 2
dropout = 0.5
eval_batch_size = 64
batch_size = 64
n_epochs = 100

device = "cuda"

criterion = nn.NLLLoss()

corpus = dataset.Corpus()


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

n_token = len(corpus.dictionary)

model = Model(
    n_token,
    input_size,
    hidden_size,
    num_layers,
    dropout
).to(device)


def get_batch(source, i):
    seq_len = min(sequence_length, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    model.eval()
    total_loss = 0.

    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, sequence_length):
            data, targets = get_batch(data_source, i)

            hidden = tuple(v.detach() for v in hidden)
            output, hidden = model(data, hidden)

            loss = criterion(output, targets)
            total_loss += loss.item() * len(data)

    return total_loss / len(data_source)


train_loss = 0
train_ppl = 0


def train():
    global train_loss
    global train_ppl

    model.train()
    total_loss = 0.
    start_time = time.time()

    hidden = model.init_hidden(batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, sequence_length)):

        data, targets = get_batch(train_data, i)
        model.zero_grad()

        hidden = tuple(v.detach() for v in hidden)
        output, hidden = model(data, hidden)

        loss = criterion(output, targets)
        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            cur_ppl = math.exp(cur_loss)
            elapsed = time.time() - start_time

            train_loss = cur_loss
            train_ppl = cur_ppl

            print('| epoch {:3d} | '
                  '{:5d}/{:5d} batches | '
                  'ms/batch {:5.2f} | '
                  'loss {:5.2f} | '
                  'ppl {:8.2f} |'
                  .format(
                      epoch,
                      batch,
                      len(train_data) // sequence_length,
                      elapsed * 1000 / log_interval,
                      cur_loss,
                      cur_ppl,
                  ))

            total_loss = 0
            start_time = time.time()

fileNumber = str(len(os.listdir("runs"))+1)
txt = open("runs/exp" + fileNumber + ".txt", 'w')
txt.write('epoch\ttrain_loss\ttrain_ppl\tval_loss\tval_ppl\n')

best_val_loss = None

opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

for epoch in range(1, n_epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(val_data)
    val_ppl = math.exp(val_loss)

    txt.write('{}\t{}\t{}\t{}\t{}\n'.format(
        epoch,
        train_loss,
        train_ppl,
        val_loss,
        val_ppl,
    ))

    print('-' * 91)
    print('| end of epoch {:3d} | '
          'time: {:5.2f}s | '
          'valid loss {:5.2f} | '
          'valid ppl {:8.2f} |'
          .format
          (
              epoch,
              (time.time() - epoch_start_time),
              val_loss,
              val_ppl,
          ))
    print('-' * 91)

    if not best_val_loss or val_loss < best_val_loss:
        with open('models/exp' + fileNumber + '.pt', 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss

with open('model.pt', 'rb') as f:
    model = torch.load(f)

test_loss = evaluate(test_data)
print('=' * 91)
print('| End of training | '
      'test loss {:5.2f} | '
      'test ppl {:8.2f} |'
      .format(
          test_loss,
          math.exp(test_loss),
      ))
print('=' * 91)

txt.close()
