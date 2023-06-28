from flask import *
import json 
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
app=Flask(__name__,template_folder='template', static_folder='static')
app.config['UPLOAD_FOLDER']= 'files'
import pdfread
import os
import torch

@app.route('/')
def hello():
    return render_template('summary.html')

@app.route('/error')
def error():
    return render_template('404.html')

@app.route('/predict', methods=['POST'])
def get_TextPrediction():
    try:
        input_json = request.get_json(force=True) 
        summary= summarizeT(input_json['text'],input_json['min_words'],input_json['max_words'])
        dictToReturn = {'summary':summary, 'max_words':input_json['max_words'],'status':200}
    except:
        dictToReturn = {'summary':'', 'max_words':0, 'status':500}

    return jsonify(dictToReturn)

@app.route('/upload', methods=['POST'])
def get_FilePrediction():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    print('uploaded successfully')
    name=(f.filename).replace(' ','_')
    ext= os.path.splitext(name)[1]
    print(ext)
    if( ext == '.docx'):
        text = pdfread.readDocx(name)
    if( ext== '.text'):
        text= pdfread.readText(name)
    if (ext== '.pdf'):
        text= pdfread.readPDF(name)
    summary=''
    try:
        summary= summarizeT(text,100,500)
        dictToReturn = {'summary':summary, 'max_words':'max_words', 'status':200}
    except:
        dictToReturn = {'summary':'', 'max_words':'max_words', 'status':500}
    print(summary)
    if os.path.exists(name):
        os.remove(name)
    else:
        print("The file does not exist")
    
    return jsonify(dictToReturn)

def summarizeP(sequence, max_length, min_length):
    torch.cuda.empty_cache()
    model_name = 'google/pegasus-xsum'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    model.config.max_position_embeddings=512
    print(model.config.max_position_embeddings)
    batch = tokenizer(sequence, truncation=True, padding='max_length', return_tensors="pt").to(device)
    translated = model.generate(**batch,min_length=180, max_new_tokens=200)
    summary = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return summary[0]
def summarizeT(sequence, min_length, max_length):
    print('input: '+sequence[:100])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer=AutoTokenizer.from_pretrained('T5-base')
    model=AutoModelForSeq2SeqLM.from_pretrained('T5-base', return_dict=True).to(device)
    inputs=tokenizer.encode("sumarize: " +sequence,return_tensors='pt', max_length=512, truncation=True).to(device)
    output = model.generate(inputs, min_length=min_length, max_length=max_length).to(device)
    summary=tokenizer.decode(output[0],skip_special_tokens=True)
    torch.cuda.empty_cache()
    return summary

if __name__ == '__main__':  
   app.run()