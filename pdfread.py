from PyPDF2 import PdfReader
import docx

def readText(inp):
    file=open(inp, mode='r')
    text = file.read()
    text= text.replace('\n','')
    return text

def readDocx(file):
    doc = docx.Document(file)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return ''.join(fullText)

# creating a pdf reader object
def readPDF(file):
    reader = PdfReader(file)
    
    # printing number of pages in pdf file
    print(len(reader.pages))
    
    text=''
    # extracting text from page
    for page in reader.pages:
        text = text+ page.extract_text()
    text=text.replace('\n', ' ')
    # print(text)
    return text
