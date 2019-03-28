import re
from nltk import sent_tokenize

with open('./0011.html', encoding='utf-8') as f:
    content = f.read()
    content = re.findall('<div.*?class.*?>(.*?)</div>', content)
    print(len(content))
    res = []
    for each in content:
        each = re.sub('<.*?>', ' ', each)
        each = re.sub('Proceedings.*', ' ', each)
        res.append(each)


a=1
