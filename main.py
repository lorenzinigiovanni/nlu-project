import os
import time
import math
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import dataset
from model import Model

# train or evaluate
is_training = True

# use the LSTM cell that I programmed
my_lstm = True

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
    not my_lstm,
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

            hidden_states = [None] * num_layers

            # make the computations
            output, _ = model(datas, lenghts, hidden_states)

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

    # compute the number of batch
    batch_num = len(train_data) // batch_size

    # cycle over all the batches
    for i in range(0, batch_num):
        datas, targets, size, lenghts = get_batch(train_data, i, batch_size)

        # set the paramenters tensors gradients to zero
        model.zero_grad()

        hidden_states = [None] * num_layers

        # make the computations
        output, _ = model(datas, lenghts, hidden_states)

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

            train_loss = cur_loss
            train_ppl = cur_ppl

            print('Epoch {:3d} | '
                  'Train loss {:5.2f} | '
                  'Train PPL {:8.2f}'
                  .format(
                      epoch,
                      cur_loss,
                      cur_ppl,
                  ))

            total_loss = 0
            total_size = 0


# generate some sentences starting from random words
def generate_sentences():
    # put the model in evaluation mode to disable dropout
    model.eval()

    with torch.no_grad():
        with open('sample.txt', 'w') as f:
            # Select one word id randomly
            rand = random.randint(1, n_token-1)
            input = torch.tensor([[rand]], device=device, dtype=torch.int64)

            sentence = []
            hidden_states = [None] * num_layers

            for _ in range(1000):
                # make the computations
                output, hidden_states = model(input, [len(sentence)+1], hidden_states)

                # sample a word id
                prob = output.exp()[-1:]
                word_id = torch.multinomial(prob, num_samples=1).item()

                # fill input with sampled word id for the next time step
                input = torch.cat((input.view(-1), torch.tensor([word_id], device=device)))
                input = input.view(-1, 1)

                word = corpus.dictionary.idx2word[word_id]
                sentence.append(word)

                if word == '<eos>':
                    # file write
                    f.write(' '.join(sentence))
                    f.write('\n')

                    # start a new sentence
                    sentence = []

                    rand = random.randint(1, n_token-1)
                    input = torch.tensor([[rand]], device=device, dtype=torch.int64)


if is_training:
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

        print()
        print('Epoch {:3d} | '
              'Valid loss {:5.2f} | '
              'Valid PPL {:8.2f}'
              .format(
                  epoch,
                  val_loss,
                  val_ppl,
              ))
        print()

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
test_ppl = math.exp(test_loss)

print()
print('Test loss {:5.2f} | '
      'Test PPL {:8.2f}'
      .format(
          test_loss,
          math.exp(test_loss),
      ))
print()

# generate sample sentences
generate_sentences()
