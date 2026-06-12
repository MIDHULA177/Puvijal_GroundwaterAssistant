import csv
from pypdf import PdfReader
import os

print('=== EXISTING CSV DISTRICTS ===')
with open('data/sih chatbot data.csv', encoding='utf-8', errors='ignore') as f:
    rows = list(csv.DictReader(f))
for r in rows:
    print(r['District'])

print('\n=== PDF CONTENT SEARCH: categorisation/exploitation ===')
for fname in ['data1.pdf','data2.pdf','data3.pdf']:
    reader = PdfReader(os.path.join('data', fname))
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or '')
        if any(w in text.lower() for w in ['over-exploit','overexploit','critical','semi-critical','safe','categoris','categori']):
            print(f'\n[{fname} p.{i+1}]')
            print(text[:800])
