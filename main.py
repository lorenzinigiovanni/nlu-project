import os
import time
import math
import torch
import torch.onnx
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import dataset
from model import Model

# train or evaluate
isTraining = False

# use the LSTM cell that I programmed
myLSTM = True

# train on cpu or cuda
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
    dropout,
    not myLSTM,
).to(device)


# get a sentence from the dataset
def get_sentence(source, i):
    # for the input of the network trim the last word
    data = source[i][:-1]
    # for the ground truth trim the first word
    target = source[i][1:]
    return data, target


# get a batch from the dataset
def get_batch(source, i, batch_size):
    datas = []
    targets = []
    lenghts = []

    for j in range(batch_size):
        data, target = get_sentence(source, i * batch_size + j)
        datas.append(data)
        targets.append(target)
        lenghts.append(len(data))

    # pad the data with the <pad> word to obtain the same length
    datas = pad_sequence(
        datas,
        padding_value=corpus.dictionary.word2idx['<pad>'],
    )
    targets = pad_sequence(
        targets,
        padding_value=corpus.dictionary.word2idx['<pad>'],
    )

    # calculate the number of words in the batch (excluding the <pad>)
    size = sum(lenghts)

    return datas, targets, size, lenghts


# calculate the loss and perplexity on validation or test data
def evaluate(data_source):
    # put the model in evaluation mode to disable dropout
    model.eval()
    total_loss = 0.
    total_size = 0

    # compute the number of batch
    batch_num = len(data_source) // eval_batch_size

    with torch.no_grad():
        # cycle over all the batches
        for i in range(0, batch_num):
            datas, targets, size, lenghts = get_batch(data_source, i, eval_batch_size)

            # initialize the hidden states
            hidden = model.init_hidden(eval_batch_size)
            # make the computations
            output, hidden = model(datas, hidden, lenghts)

            # compute the negative log likelihood loss excluding the <pad> word
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


# train the model using the training data
def train():
    global train_loss
    global train_ppl

    # put the model in training mode to enable dropout
    model.train()
    total_loss = 0.
    total_size = 0

    start_time = time.time()

    # compute the number of batch
    batch_num = len(train_data) // batch_size

    # cycle over all the batches
    for i in range(0, batch_num):
        datas, targets, size, lenghts = get_batch(train_data, i, batch_size)

        # set the paramenters tensors gradients to zero
        model.zero_grad()

        # initialize the hidden states
        hidden = model.init_hidden(batch_size)
        # make the computations
        output, hidden = model(datas, hidden, lenghts)

        # compute the negative log likelihood loss excluding the <pad> word
        loss = F.nll_loss(
            output,
            targets.view(-1),
            reduction='sum',
            ignore_index=corpus.dictionary.word2idx['<pad>'],
        )
        # compute the paramenters tensors gradients
        loss.backward()

        # clip the gradient norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # update the paramenters tensors
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


# generate a sentence starting from a random word
def generateSentence():
    # put the model in evaluation mode to disable dropout
    model.eval()

    with torch.no_grad():
        with open('sample.txt', 'w') as f:
           # initialize the hidden states
            hidden = model.init_hidden(1)

            # Select one word id randomly
            probability = torch.ones(n_token)
            input = torch.multinomial(probability, num_samples=1).unsqueeze(1).to(device)

            for _ in range(1000):
                # make the computations
                output, hidden = model(input, hidden, [1])

                # sample a word id
                prob = output.exp()[-1:]
                word_id = torch.multinomial(prob, num_samples=1).item()

                # fill input with sampled word id for the next time step
                input = torch.cat((input.view(-1), torch.tensor([word_id], device=device)))
                input = input.view(-1, 1)

                word = corpus.dictionary.idx2word[word_id]
                if word == '<eos>':
                    # start a new sentence
                    word = '\n'
                    input = torch.multinomial(probability, num_samples=1).unsqueeze(1).to(device)
                else:
                    # continue the previous sentence
                    word = word + ' '

                # file write
                f.write(word)


if isTraining:
    # store loss and ppl of the runs in txt files
    fileNumber = str(len(os.listdir("runs"))+1)
    txt = open("runs/exp" + fileNumber + ".txt", 'w')
    txt.write('epoch\ttrain_loss\ttrain_ppl\tval_loss\tval_ppl\n')

    best_val_loss = None

    # instantiate and Adam optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

    # cycle for the n_epochs
    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()

        # call the train function
        train()

        # compute the validation loss and ppl
        val_loss = evaluate(val_data)
        val_ppl = math.exp(val_loss)

        # save loss and ppl in a txt file
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

        # save the model if it is better than the previous one
        if not best_val_loss or val_loss < best_val_loss:
            with open('models/exp' + fileNumber + '.pt', 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss

    txt.close()

with open('model.pt', 'rb') as f:
    model = torch.load(f)

# compute the test loss and ppl
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

# generate a sample sentence
generateSentence()
