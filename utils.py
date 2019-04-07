import torch
from torch import nn
from random import sample
import sentencepiece as spm

class RNNModel(nn.Module):

    def __init__(self, rnn_type, ntoken, emsize, nhid, nlayers, dropout, tie_weights):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, emsize)
        assert (rnn_type in ['LSTM', 'GRU']), "Arg `rnn_type` has to be one of {GRU, LSTM}."
        self.rnn = getattr(nn, rnn_type)(emsize, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != emsize:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
             
def is_unbalanced(s):
  if s.count('"') % 2 != 0 or s.count('(') != s.count(')'):
    return True
  else:
    return False

def sample_punkt():
  return sample(['.', '?', '!'], 1)[0]

def parse_last_line(s):
  l = list(s)
  
  if l[-1] == ',': 
    l[-1] = sample_punkt()
  
  if l[-1] not in list('.?!'): 
    l.append(sample_punkt())  
  return "".join(l)
  
def generate_line(model, hidden=None, temp=1.0, 
               sos_id=1, eos_id=2, unk_id=0, max_len=15):
  """Generate line from `model` with `hidden` state at `temp`."""
  ids = []
  
  if hidden is None:
    hidden = model.init_hidden(1)
  
  input = torch.tensor([sos_id], dtype=torch.long).reshape(1,1)
  id = 0
  while id != eos_id and len(ids)<max_len :
    output, hidden = model(input, hidden)
    probs = output.squeeze().div(temp).exp()
    id = torch.multinomial(probs, num_samples=1).item() 
    if id == sos_id or id == unk_id: continue
    input.fill_(id)
    ids += [id]
  
  return ids, hidden

def get_tokenizer(path):
  proc = spm.SentencePieceProcessor()
  proc.Load(path)
  return lambda l: proc.DecodeIds(l)

def get_model(path):
  dict = torch.load(path, map_location=torch.device('cpu'))
  state, params = dict['state_dict'], dict['params']
  model = RNNModel(*params.values())
  model.load_state_dict(state)
  return model