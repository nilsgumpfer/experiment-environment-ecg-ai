from urllib.request import urlopen
from urllib.parse import quote
import json

baseUrl = 'https://snowstorm.test-nictiz.nl'


# Prints fsn of a concept
def getConceptById(id):
    url = baseUrl + '/browser/MAIN/concepts/' + id
    response = urlopen(url).read()
    data = json.loads(response.decode('utf-8'))

    return data['fsn']['term']



