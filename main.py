import os
import time
import math
import torch
import torch.nn as nn
import torch.onnx
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import dataset
from model import Model

isTraining = True
device = "cuda"

clip = 0.25
log_interval = 50
input_size = 512
hidden_size = 512
num_layers = 2
dropout = 0.5
eval_batch_size = 1
batch_size = 64
n_epochs = 100

corpus = dataset.Corpus(device)

train_data = corpus.train
val_data = corpus.valid
test_data = corpus.test

n_token = len(corpus.dictionary)

model = Model(
    n_token,
    input_size,
    hidden_size,
    num_layers,
    dropout
).to(device)


def get_sentence(source, i):
    data = source[i][:-1]  # .view(-1, 1)
    target = source[i][1:]
    return data, target


def get_batch(source, i, batch_size):
    datas = []
    targets = []
    lenghts = []

    for j in range(batch_size):  # seq_len = min(args.bptt, len(source) - 1 - i)
        data, target = get_sentence(source, i * batch_size + j)
        datas.append(data)
        targets.append(target)
        lenghts.append(len(data))

    datas = pad_sequence(
        datas, padding_value=corpus.dictionary.word2idx['<pad>'])
    targets = pad_sequence(
        targets, padding_value=corpus.dictionary.word2idx['<pad>'])

    size = sum(lenghts)

    return datas, targets, size


def evaluate(data_source):
    model.eval()
    total_loss = 0.
    total_size = 0

    hidden = model.init_hidden(eval_batch_size)

    batch_num = len(data_source) // eval_batch_size

    with torch.no_grad():
        for i in range(0, batch_num):
            datas, targets, size = get_batch(train_data, i, eval_batch_size)

            # hidden = tuple(v.detach() for v in hidden)
            hidden = model.init_hidden(eval_batch_size)
            output, hidden = model(datas, hidden)

            loss = F.nll_loss(
                output,
                targets.view(-1),
                reduction='sum',
                ignore_index=corpus.dictionary.word2idx['<pad>'],
            )

            total_loss += loss.item()
            total_size += size

    return total_loss / total_size


train_loss = 0
train_ppl = 0


def train():
    global train_loss
    global train_ppl

    model.train()
    total_loss = 0.
    total_size = 0

    start_time = time.time()

    hidden = model.init_hidden(batch_size)

    batch_num = len(train_data) // batch_size

    for i in range(0, batch_num):
        datas, targets, size = get_batch(train_data, i, batch_size)

        model.zero_grad()

        # hidden = tuple(v.detach() for v in hidden)
        hidden = model.init_hidden(batch_size)
        output, hidden = model(datas, hidden)

        # output = target len * batch size

        # output = pack_padded_sequence(output, lenghts, enforce_sorted=False)

        loss = F.nll_loss(
            output,
            targets.view(-1),
            reduction='sum',
            ignore_index=corpus.dictionary.word2idx['<pad>'],
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        total_loss += loss.item()
        total_size += size

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / total_size
            cur_ppl = math.exp(cur_loss)
            elapsed = time.time() - start_time

            train_loss = cur_loss
            train_ppl = cur_ppl

            print('| epoch {:3d} | '
                  '{:5d}/{:5d} batch | '
                  'ms/batch {:5.2f} | '
                  'loss {:5.2f} | '
                  'ppl {:8.2f} |'
                  .format(
                      epoch,
                      i,
                      batch_num,
                      elapsed * 1000 / log_interval,
                      cur_loss,
                      cur_ppl,
                  ))

            total_loss = 0
            total_size = 0
            start_time = time.time()


if isTraining:
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

# vesuvio()

if isTraining:
    txt.close()
