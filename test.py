


# import spacy

# texts = [
#     "LAWRENCE L DEVINE Honolulu, HI 96820-0260 MMOH LLC",
#     "10/1/2023 LAWRENCE DEVINE Alert Alarm Hawaii",
#     "Auto Pay Scheduled - Do Not Send Payment 00005912 01 AB 0.537 KG091711 0022 XX PHILEMON EQUITIES LLC",
#     "Glen Breagha LLC",
#     "Customer ID Group Number: LAWRENCE L DEVINE",
#     "TR00003 LAWRENCE L DEVINE NO"
# ]

# nlp = spacy.load("en_core_web_sm")

# customer_names = []

# for text in texts:
#     doc = nlp(text)
#     for ent in doc.ents:
#         if ent.label_ == "PERSON":
#             customer_names.append(ent.text)

# # Print the extracted customer names
# for i, name in enumerate(customer_names, start=1):
#     print(f"t{i}: {name}")
# SECOND WAY USING Spacy

try:
    import spacy
    import json
except Exception as e:
    print(e)
    

class EntityGenerator(object):
    
    _slots__ = ['text']
    
    def __init__(self, text=None):
        self.text = text
        
    def get(self):
        """
        Return a Json
        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self.text)
        text = [ent.text for ent in doc.ents]
        entity = [ent.label_ for ent in doc.ents]
    
        from collections import Counter
        import json

        data = Counter(zip(entity))
        unique_entity = list(data.keys())
        unique_entity = [x[0] for x in unique_entity]

        d = {}
        for val in unique_entity:
            d[val] = []

        for key,val in dict(zip(text, entity)).items():
            if val in unique_entity:
                d[val].append(key)
        return d


try:
    import PyPDF2
    import requests
    import json
except Exception:
    pass

class Resume(object):
    def __init__(self, filename=None):
        self.filename = filename
        
    def get(self):
        """
        
        """
        fFileObj = open(self.filename, 'rb')
        pdfReader = PyPDF2.PdfFileReader(fFileObj)
        pageObj = pdfReader.getPage(0)
        print("Total Pages : {} ".format(pdfReader.numPages))

        resume = pageObj.extractText()
        return resume

resume = Resume(filename="0.pdf")
response_news = resume.get()
helper = EntityGenerator(text=response_news)
response = helper.get()
print(json.dumps(response , indent=3))