from pyspark import SparkContext
import csv
import pandas as pd


symptom_list=["pain","pressure","squeezing","chest pain","discomfort","shortness of breath","cold sweat","nausea","lightheadedness",\
"arm weaknes","dizziness","loss of vision","confusion","severe headache","bleeding","itching","burning","frequent urination"\
,"nipple discharge","breast tenderness","skin changes","cough","pneumonia","wheezing","constipation","diarrhea"\
,"vomiting","flushing","redness","jaundice","muscle pain","numbness","stiffness","swelling","inflammation","anxiety","depression"\
,"nightmares","hallucinations","starvation","dehydration","runny nose"]

symptons={}
i=0
for s in symptom_list:
    symptons[s]=i
    i+=1

# print(symptons)

def split_tweet_and_location_with_date(x):
    x=x.split('\t')
    # return as tweet, location, date
    if len(x)==2:
        return x[0], '', x[1].split('-')[1]
    elif(len(x)==3):
        return x[0], x[1], x[2].split('-')[1]
    else:
        return x[0], "",""

def map_symptons_to_locations_and_year_month(x):
    l=[0 for _ in range(len(symptons))]
    for word in x[0].split(' '):
        if word in symptons:
            l[symptons[word]]+=1
    return (x[1], x[2]), l

def reduce_symptons_to_location(x):
    return x[0], [sum(x) for x in zip(*x[1])]

sc = SparkContext("local", "project")
tweets = sc.textFile("filteredtweets.txt")
tweets_with_location=tweets.map(split_tweet_and_location_with_date)
symptons_with_locations_and_year_month=tweets_with_location.map(map_symptons_to_locations_and_year_month)
symptons_with_locations_and_year_month=symptons_with_locations_and_year_month.groupByKey()
symptons_with_locations=symptons_with_locations_and_year_month.map(reduce_symptons_to_location)


#Analysys for texas
symptons_for_texas=symptons_with_locations.filter(lambda x: x[0][0]=='TX')

#dehydration, nausea, chest pain, flushing, cold sweat, pain
symptons_for_texas_for_influenza=symptons_for_texas.map(lambda x: (x[0], [x[1][40], x[1][7], x[1][3], x[1][27], x[1][6]]))

#vomiting, urin, jaundice, diarehha, itching, muscle pain
symptons_for_texas_for_hepatatisA=symptons_for_texas.map(lambda x: (x[0], [x[1][26], x[1][17], x[1][29], x[1][25], x[1][15], x[1][30]]))


s=0
# forming influenza matrix that is used for linear regression and symptoms are our coloumns
influenza=[]
for i in symptons_for_texas_for_influenza.collect():
    influenza.append([i[0][0], i[0][1]]+i[1])
influenza.sort(key=lambda x:int(x[1]))
column_names=["state", "week", "dehydration","cough",  "chest pain", "runny nose", "cold sweat"]
res=pd.DataFrame(influenza, columns=column_names)
# res.to_csv("influenza_count_matrix.csv", sep=",", index=False)
print(res)


# forming hepatitis A matrix that is used for linear regression and symptoms are our coloumns
hepatitisA=[]
for i in symptons_for_texas_for_hepatatisA.collect():
    hepatitisA.append([i[0][0], i[0][1]]+i[1])
hepatitisA.sort(key=lambda x:int(x[1]))
column_names=["state", "week", "vomiting", "urine", "jaundice", "diarehha", "itching", "muscle pain"]
res=pd.DataFrame(hepatitisA, columns=column_names)
# res.to_csv("hepatitisA_count_matrix.csv", sep=",", index=False)
print(res)
