from flask import *
import json 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
app=Flask(__name__,template_folder='template', static_folder='static')


@app.route('/')
def hello():
    return render_template('summary.html')

@app.route('/predict', methods=['POST'])
def get_students():
    input_json = request.get_json(force=True) 
    summary= summarize(input_json['text'],input_json['max_words'],input_json['min_words'])
    dictToReturn = {'summary':summary, 'max_words':input_json['max_words']}
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