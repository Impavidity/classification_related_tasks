import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable

class RNNCNN(nn.Module):
    def __init__(self, config):
        super(RNNCNN, self).__init__()
        Ks = 1 # There are three conv net here
        input_channel = 1
        self.config = config
        self.embed = nn.Embedding(config.words_num, config.words_dim)
        self.static_embed = nn.Embedding(config.embed_num, config.embed_dim)
        self.non_static_embed = nn.Embedding(config.embed_num, config.embed_dim)
        self.static_embed.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size=config.lstm_input,
                           hidden_size=config.lstm_hidden,
                           num_layers=config.lstm_layer,
                           dropout=config.lstm_dropout,
                           bidirectional=True)
        kernel_size = config.lstm_hidden * 2
        self.conv1 = nn.Conv2d(input_channel, config.output_channel, (1, kernel_size))
        #self.conv2 = nn.Conv2d(input_channel, config.output_channel, (4, kernel_size), padding=(3,0))
        #self.conv3 = nn.Conv2d(input_channel, config.output_channel, (5, kernel_size), padding=(4,0))

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(Ks * config.output_channel +  2 * config.lstm_hidden, config.linear_hidden)
        self.fc2 = nn.Linear(config.linear_hidden, config.target_class)


    def forward(self, x):
        # x = (sequence length, batch_size, dimension of embedding)
        text = x.text
        batch_size = text.size()[1]
        x = self.non_static_embed(text)
        # h0 / c0 = (layer*direction, batch_size, hidden_dim)
        if self.config.cuda:
            h0 = Variable(torch.zeros(self.config.lstm_layer * 2, batch_size,
                                      self.config.lstm_hidden).cuda())
            c0 = Variable(torch.zeros(self.config.lstm_layer * 2, batch_size,
                                      self.config.lstm_hidden).cuda())
        else:
            h0 = Variable(torch.zeros(self.config.lstm_layer * 2, batch_size,
                                      self.config.lstm_hidden))
            c0 = Variable(torch.zeros(self.config.lstm_layer * 2, batch_size,
                                      self.config.lstm_hidden))
        # output = (sentence length, batch_size, hidden_size * num_direction)
        # ht = (layer*direction, batch, hidden_dim)
        # ct = (layer*direction, batch, hidden_dim)
        outputs, (ht, ct) = self.lstm(x, (h0, c0))
        lstm_hidden = ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        # lstm_hidden = (batch, layer*direction * hidden_dim)
        lstm_outputs = outputs.transpose(0, 1).contiguous()
        x = lstm_outputs.unsqueeze(1)
        #x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        x = [F.relu(self.conv1(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * Ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # max-over-time pooling
        # (batch, channel_output) * Ks
        x.append(lstm_hidden)
        x = torch.cat(x, 1) # (batch, channel_output * Ks)
        x = self.dropout(x)
        x = F.relu(self.fc1(x)) # (batch, target_size)
        logit = self.fc2(x)
        return logit
