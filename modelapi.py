from flask import *
import json 
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
app=Flask(__name__,template_folder='template', static_folder='static')
app.config['UPLOAD_FOLDER']= 'files'
import pdfread
import os

@app.route('/')
def hello():
    return render_template('summary.html')

@app.route('/predict', methods=['POST'])
def get_TextPrediction():
    input_json = request.get_json(force=True) 
    summary= summarize(input_json['text'],input_json['max_words'],input_json['min_words'])
    dictToReturn = {'summary':summary, 'max_words':input_json['max_words']}
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
    summary= summarize(text,500,100)
    print(summary)
    if os.path.exists(name):
        os.remove(name)
    else:
        print("The file does not exist")
    dictToReturn = {'summary':summary, 'max_words':'max_words'}
    return jsonify(dictToReturn)

def summarize(sequence, max_length, min_length):
    tokenizer=AutoTokenizer.from_pretrained('T5-base')
    model=AutoModelForSeq2SeqLM.from_pretrained('T5-base', return_dict=True) 
    inputs=tokenizer.encode("sumarize: " +sequence,return_tensors='pt', max_length=512, truncation=True)
    output = model.generate(inputs, min_length=int(min_length), max_length=int(max_length))
    summary=tokenizer.decode(output[0],skip_special_tokens=True)
    return summary
if __name__ == '__main__':  
   app.run()