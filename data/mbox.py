import re

import pandas as pd
from bs4 import BeautifulSoup

INTERESTED_HEADERS = [
    'To',
    'Subject',
    'Date',
    'From',
]

PATTERNS = {
    'boundary' :'boundary="(.*)"',
    'charset': 'charset="?(.*)"?',
    'content-transfer-encoding': 'Content-Transfer-Encoding: (.*)',
    'date': 'Date: (.*)',
    'from' : 'From: (.*)',
    'to' : 'To: (.*)',
    'subject' : 'Subject: (.*)'
}

with open("1577894392.3457045_5.txt", 'r') as f:
    text = f.read()
    email = {}
    for field, pattern in PATTERNS.items():
        email[field] = re.findall(pattern, text)[0]
            
    text = text.split("--" + email['boundary'])[1:]
    for i in range(len(text)):
        temp = text[i].split("\n\n", 1)
        if re.findall('Content-Type: text/(.*);', temp[0])[0] == "plain":
            text[i] = temp
        elif re.findall('Content-Type: text/(.*);', temp[0])[0] == 'html':
            soup = BeautifulSoup(temp[1:], 'html.parser')
            text[i] = soup.get_text()

    text = list(filter(None, text))
    email['message'] = text[0]
    df = pd.DataFrame([email])

df.to_csv("Email.csv")