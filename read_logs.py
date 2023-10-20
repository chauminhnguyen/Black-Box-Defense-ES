import os
import re

def read_logs(path):
    with open(path, 'r') as f:
        data = f.read()
    epoches = data.split('\n20it ')
    
    losses = []
    acc1 = []
    acc5 = []
    for epoch in epoches:
        last_line = re.findall('Test: \[.+', epoch)
        if len(last_line) < 1:
            continue
        last_line = last_line[-1]
        # losses.append(re.findall('Loss \d+.\d+ \(\d+.\d+\)', last_line))
        # acc1.append(re.findall('Acc@1 \d+.\d+ \(\d+.\d+\)', last_line))
        # acc5.append(re.findall('Acc@5 \d+.\d+ \(\d+.\d+\)', last_line))
        match = re.search("Loss \d+\.\d+ \((\d+\.\d+)\)\s+Acc@1 \d+\.\d+ \((\d+\.\d+)\)\s+Acc@5 \d+\.\d+ \((\d+\.\d+)\)", last_line)
        if match:
            losses.append(match.group(1))  # Extract the captured value
            acc1.append(match.group(2))
            acc5.append(match.group(3))
            # print("Loss:", loss_value, acc1_value, acc5_value)
        else:
            print("Loss not found in the text.")

    print(losses)
    print(acc1)
    print(acc5)
read_logs('logs/remote_logs.txt')