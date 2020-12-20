# Rebecca Byun

import numpy as np
import matplotlib.pyplot as plt
import csv

'''
def printResult(k, digit):
  for x in k:
    print("-------------------- k = %d --------------------" % x)
    for d in digit:
      print("digit = %d,\tright = %d,\tleft = %d" % (d,d,d))
    print("\nTotals:\t\tright = %d,\twrong = %d,\t%.2f%\n")
'''

def trainSVD(A):
  U, s, V = np.linalg.svd(A)
  return U

def compareClass(k, train, test, k_norm):
  k_norm.clear()
  for x in k:
    U = train[:,0:x]
    U_T = U.T
    I = np.identity(np.size(U, 0))
    for y in range(100):
      k_norm.append(np.linalg.norm((I - (U).dot(U_T)).dot(test[:, y]), 2))
  return k_norm

if __name__ == "__main__":
  train = []
  test = []
  k = np.array([1,5,20,100,256])
  digit = np.arange(10)
  norm = []
  k_norm = []

  with open('digitsTrainCombined.csv', mode='r') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    count = np.arange(4000)
    for row in reader:
      for x in count:
        train.append(float(row[x]))

  with open('digitsTestCombined.csv', mode='r') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    count = np.arange(1000)
    for row in reader:
      for x in count:
        test.append(float(row[x]))

  train = np.array(train)
  train = np.reshape(train, (256,4000))
  test = np.array(test)
  test = np.reshape(test, (256,1000))

  train_0 = train[:,    0: 400] # training samples
  train_1 = train[:,  400: 800]
  train_2 = train[:,  800:1200]
  train_3 = train[:, 1200:1600]
  train_4 = train[:, 1600:2000]
  train_5 = train[:, 2000:2400]
  train_6 = train[:, 2400:2800]
  train_7 = train[:, 2800:3200]
  train_8 = train[:, 3200:3600]
  train_9 = train[:, 3600:4000]

  test_0 = test[:,   0: 100]  # test samples
  test_1 = test[:, 100: 200]
  test_2 = test[:, 200: 300]
  test_3 = test[:, 300: 400]
  test_4 = test[:, 400: 500]
  test_5 = test[:, 500: 600]
  test_6 = test[:, 600: 700]
  test_7 = test[:, 700: 800]
  test_8 = test[:, 800: 900]
  test_9 = test[:, 900:1000]

  train_U_0 = trainSVD(train_0)
  train_U_1 = trainSVD(train_1)
  train_U_2 = trainSVD(train_2)
  train_U_3 = trainSVD(train_3)
  train_U_4 = trainSVD(train_4)
  train_U_5 = trainSVD(train_5)
  train_U_6 = trainSVD(train_6)
  train_U_7 = trainSVD(train_7)
  train_U_8 = trainSVD(train_8)
  train_U_9 = trainSVD(train_9)

  k_norm = compareClass(k, train_U_0, test_0, k_norm)
  norm = np.array(k_norm).T
  print(np.shape(norm))
  k_norm = compareClass(k, train_U_1, test_1, k_norm)
  print(np.shape(norm))
  k_norm = compareClass(k, train_U_2, test_2, k_norm)

  k_norm = compareClass(k, train_U_3, test_3, k_norm)

  k_norm = compareClass(k, train_U_4, test_4, k_norm)
  
  k_norm = compareClass(k, train_U_5, test_5, k_norm)
  
  k_norm = compareClass(k, train_U_6, test_6, k_norm)
  
  k_norm = compareClass(k, train_U_7, test_7, k_norm)
  
  k_norm = compareClass(k, train_U_8, test_8, k_norm)
  
  k_norm = compareClass(k, train_U_9, test_9, k_norm)

  print(np.shape(norm))