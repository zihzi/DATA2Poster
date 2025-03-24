import pandas as pd

def preprocess(dataList):
    lenN = dataList.columns.size
    NumFall = [0] * lenN
    ValueFall = []
    for j in range(lenN):
        ValueSingle = []
        for i in range(len(dataList)):
            if dataList.iloc[i,j] not in ValueSingle:
                ValueSingle.append(dataList.iloc[i,j])
        ValueFall.append(ValueSingle)
        NumFall[j] = len(ValueSingle) 
    return ValueFall, NumFall  
# print("ValueFall: ",ValueFall)
# print("NumFall: ",NumFall)

def count(list,data):
    number = 0
    for i in range(len(data)):
        isIn = True
        for j in range(len(list)):
            if list[j] not in data.iloc[i].to_list():
                isIn = False
                break
        if isIn:
            number += 1
    return number
def BUC(dataList, tempList, n, curN, min_sup = 50):
    ValueFall, NumFall = preprocess(dataList)
    if curN == n:
        return
    for i in range(NumFall[curN]):
        tempList.append(ValueFall[curN][i])
        if count(tempList, dataList) >= min_sup:
            if len(tempList) <= 2:
             print("%s: %d" % (tempList, count(tempList, dataList)))
             BUC(dataList,tempList, n, curN+1)
        tempList.pop()
    BUC(dataList,tempList, n, curN+1)
