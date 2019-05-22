from pyspark import SparkContext
import csv
import pandas as pd


sc = SparkContext("local", "project")
tweets = sc.textFile("hospital_data.tsv").map(lambda x: x.split('\t'))
'''
l=tweets.take(2)[1]
for i, j in enumerate(l):
    print(i, " ", j)
'''
#filtering out the diseases (covering 2 diseases)-influenza and hepatitis A
tweets_filtered=tweets.filter(lambda x: x[34]=='influenza' or x[34]=='hepatitis A')
tweets_filtered=tweets_filtered.map(lambda x: ((x[1].split("-")[1], x[34]), 1))
tweets_filtered=tweets_filtered.groupByKey()
counts=tweets_filtered.map(lambda x: (x[0], sum(x[1])))


inf=[]
counts_influenxa=counts.filter(lambda x:x[0][1]=="influenza")
for i in counts_influenxa.collect():
    inf.append([i[0][1], i[0][0], i[1]])
inf.sort(key=lambda x:int(x[1]))
print(inf)
res=pd.DataFrame(inf, columns=["disease", "week", "y"])
# res.to_csv("influenza_hospital.csv", sep=",", index=False)
print(res)



hepA=[]
counts_hep=counts.filter(lambda x:x[0][1]=="hepatitis A")
for i in counts_hep.collect():
    hepA.append([i[0][1], i[0][0], i[1]])
hepA.sort(key=lambda x:int(x[1]))
print(hepA)
res=pd.DataFrame(hepA, columns=["disease", "week", "y"])
# res.to_csv("hepatitisA_hospital.csv", sep=",", index=False)
print(res)
