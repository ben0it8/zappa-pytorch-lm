from random import sample
from flask import Flask, Response, json
from flask_cors import CORS
import os
import pickle
import requests
import torch

from utils import RNNModel, generate_line, parse_last_line, is_unbalanced, get_model, get_tokenizer

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'models/model_state.pth'
TOKENIZER_PATH = 'models/tokenizer.model'

def init():
  global model
  global tokenizer
  model = get_model(MODEL_PATH)
  tokenizer = get_tokenizer(TOKENIZER_PATH)

def generate_text(model, tokenizer, num_lines=10, min_len=8, max_len=15,
             unk_id=0, sos_id=1, eos_id=2, temp=0.55):

  model.eval()  
  lines = []
  line_cnt = 0
  hidden = model.init_hidden(1)
  with torch.no_grad():
    
    while line_cnt != num_lines:
      try:
        ids, hidden = generate_line(model, hidden=hidden, temp=temp, max_len=max_len,
                                    sos_id=sos_id, eos_id=eos_id, unk_id=unk_id)
        
        if len(ids) <= min_len: raise Exception
        line = tokenizer(ids).strip()
        
        if line.startswith(tuple("-?!.,()")): raise Exception
        if is_unbalanced(line): raise Exception
        
        lines += [line]
        line_cnt +=1
        
      except: pass
    
  last_line = lines.pop()
  l = parse_last_line(last_line)
  lines.append(l)
  
  text = "\n".join(lines)
  json_resp = json.dumps({'response': text}, ensure_ascii=False)
  return Response(response=json_resp, status=200, mimetype='application/json')

@app.route('/generate', methods=['GET'])
def generate():
  return generate_text(model, tokenizer)

if __name__ == "__main__":
    print("Loading model files...")
    init()
    app.run()
