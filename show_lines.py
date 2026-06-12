lines = open('app.py', encoding='utf-8').readlines()
out = open('lines_out.txt', 'w', encoding='utf-8')
for i, l in enumerate(lines[318:342], start=319):
    out.write(str(i) + ' ' + repr(l) + '\n')
out.close()
print('done')
