import csv
val = "./r=0/average/average-time.csv"
arr = []
res = []
with open(val) as f:
  reader = csv.reader(f)
  for row in reader:
    arr.append(row)

with open('average.csv', 'w') as f:
  writer = csv.writer(f)
  for i in range(len(arr[0])):
    a = 0
    for k in range(len(arr)):
      a += float(arr[k][i])
    writer.writerow([i+1, a/len(arr)])
