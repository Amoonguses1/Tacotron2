import csv


with open("download/LJSpeech-1.1/metadata.csv") as f:
    reader = csv.reader(f, delimiter="|")
    data = [row for row in reader]
    train, dev, eval = [], [], []
for i, row in enumerate(data):
    if i < 11900:
        train.append(row[0])
    elif i < 12700:
        dev.append(row[0])
    else:
        eval.append(row[0])
    with open("download/LJSpeech-1.1/txt/"+row[0]+".txt", 'w') as f:
        f.write(row[1])
with open("data/train.list", 'w') as f:
    for st in train:
        f.write(st+"\n")
with open("data/dev.list", 'w') as f:
    for st in dev:
        f.write(st+"\n")
with open("data/eval.list", 'w') as f:
    for st in eval:
        f.write(st+"\n")