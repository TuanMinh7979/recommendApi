import re
import numpy as np
from string import punctuation
from pymongo import MongoClient
import pymongo
from django.http import JsonResponse
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyvi import ViTokenizer
import os
dirname = os.path.dirname(__file__)
module_dir = os.path.dirname(__file__)  # get current directory
# cvFilePath = os.path.join(module_dir, '../media/cvsDbData1.xlsx',)
# jobFilePath = os.path.join(module_dir, '../media/jobsDbData1.xlsx',)
cvFilePath = 'D:/sugDjango/recommendApi/media/cvsDbData1.xlsx'
jobFilePath = "D:/sugDjango/recommendApi/media/jobDbData1.xlsx"


def cleanhtml(raw_html):
    cleantext = ''
    try:
        raw_html = raw_html.replace(" br ", " ")
        raw_html = raw_html.replace("nbsp", " ")
        raw_html = raw_html.replace("ndash", " ")
        raw_html = raw_html.replace("&rsquo;", ' ')
        raw_html = raw_html.replace("&trade;", ' ')
        raw_html = raw_html.replace("&amp", ' ')
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', raw_html)
        cleantext = ' '.join(cleantext.split())
    except:
        print(raw_html)
    return cleantext


def text_preprocessing(text):
    text = text.lower()
    text = text.translate(str.maketrans(' ', ' ', punctuation))
    text = ' '.join(text.split())
    return text


def getStopWord():
    stop_word = []
    with open(os.path.join(module_dir, '../media/viStop.txt',), encoding="utf-8") as f:
        text = f.read()
        for word in text.split():
            stop_word.append(word)
        f.close()
    stop_word = stop_word
    return (stop_word)


def rmStw(text, stop_word):
    try:
        rs = []
        for word in text.split(" "):
            if (word not in stop_word):
                if ("_" in word) or (word.isalpha() == True):
                    rs.append(word)
    except:
        print(rs)

    return " ".join(rs)

# tokenize


def tokenize(text):
    try:
        doc = ViTokenizer.tokenize(text)
        return doc
    except:
        print(doc)

    return doc

def updateJobsFile(request):
    client = MongoClient()
    # point the client at mongo URI
    client = MongoClient(
        'mongodb+srv://tuan:12345678Abc@cluster0.h8bya9k.mongodb.net/')
    # #select database
    db = client['jobapp']
    # #select the collection within the database
    jobposts = db.jobposts
    jobpostsDb = pd.DataFrame(list(jobposts.find()))
    jobpostsDb = jobpostsDb.fillna('')
    jobpostsDb['fulltext'] = jobpostsDb['candidateRequiredText'] + \
        jobpostsDb['descriptionText']+jobpostsDb['title']
    jobs = jobpostsDb[["_id", "fulltext"]]
    jobs['fulltext'] = np.vectorize(cleanhtml)(jobs['fulltext'])
    jobs['fulltext'] = np.vectorize(text_preprocessing)(jobs['fulltext'])
    jobs["fulltext"]=jobs["fulltext"].apply(lambda x: tokenize(x))
    stw = getStopWord()
    jobs["fulltext"]=jobs["fulltext"].apply(lambda x: rmStw(x, stw))
   
    jobs.to_excel(jobFilePath, index=True)
    return JsonResponse(status=200, data={"message":"updated jobs success"})


def updateCvsFile(request):
    client = MongoClient()
    # point the client at mongo URI
    client = MongoClient(
        'mongodb+srv://tuan:12345678Abc@cluster0.h8bya9k.mongodb.net/')
    # #select database
    db = client['jobapp']
    # #select the collection within the database
    resume = db.resumes
    resumeDb = pd.DataFrame(list(resume.find()))

   
    resumeDb = resumeDb.fillna('')
    
    resumeDb['fulltext'] = resumeDb['experience']+resumeDb['skills']

    #
   
    cvs = resumeDb[["_id", "fulltext"]]
    # text preprocessing, remove html, punctuation
   
    cvs['fulltext'] = np.vectorize(cleanhtml)(cvs['fulltext'])
    cvs['fulltext'] = np.vectorize(text_preprocessing)(cvs['fulltext'])
    #tokenize
    
    cvs["fulltext"]=cvs["fulltext"].apply(lambda x: tokenize(x))
    # remove stopword
    stw = getStopWord()
    
    cvs["fulltext"]=cvs["fulltext"].apply(lambda x: rmStw(x, stw))

   
  
    
    cvs.to_excel(cvFilePath, index=True)

    return JsonResponse(status=200, data={"message":"updated cvs success"})

def getSugCvForJob(request, jobId):

    cvdf = pd.read_excel(cvFilePath)
    jobdf = pd.read_excel(jobFilePath)

    givenJobRow = jobdf.loc[jobdf['_id'] == jobId]


    ls = []
    ls.append(givenJobRow['fulltext'].tolist()[0])
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_jobs = tfidf_vectorizer.fit_transform(ls)

    tfidf_cvs = tfidf_vectorizer.transform(cvdf['fulltext'])

    cos_similarity = map(lambda x: cosine_similarity(tfidf_cvs, x), tfidf_jobs)
    simrs = list(cos_similarity)

    rs = simrs[0]
    kq = { str(idx):x[0] for idx, x in enumerate(rs)}
   
    srs = sorted(kq.items(), key=lambda x:x[1])
    fiveBestSimilar = srs[-5:]
   
    finalRs = [int(x[0]) for x in fiveBestSimilar]
    ids = []
    for idx in finalRs:
        ids.append(cvdf.iloc[idx]['_id'])
    print(ids)

    return JsonResponse(status=200, data={"sugList": ids})


def getSugJobForCv(request, cvId):

    cvdf = pd.read_excel(cvFilePath)
    jobdf = pd.read_excel(jobFilePath)

    givenCvRow = cvdf.loc[cvdf['_id'] == cvId]

    

    ls = []
    ls.append(givenCvRow['fulltext'].tolist()[0])

    tfidf_vectorizer1 = TfidfVectorizer()
    tfidf_cvs = tfidf_vectorizer1.fit_transform(ls)
    tfidf_jobs = tfidf_vectorizer1.transform(jobdf['fulltext'])
    cos_similarity = map(lambda x: cosine_similarity(tfidf_jobs, x), tfidf_cvs)
    simrs = list(cos_similarity)

    rs = simrs[0]
    kq = { str(idx):x[0] for idx, x in enumerate(rs)}

    srs = sorted(kq.items(), key=lambda x:x[1])

    fiveBestSimilar = srs[-5:]
   
    finalRs = [int(x[0]) for x in fiveBestSimilar]
    ids = []
    for idx in finalRs:
        ids.append(jobdf.iloc[idx]['_id'])
    return JsonResponse(status=200, data={"sugList": ids})


def getSimilarJob(request, jobId):
    jobdf = pd.read_excel(jobFilePath)

    jobToFindSim = jobdf.loc[jobdf['_id'] == jobId]
    tfidf_vectorizer2 = TfidfVectorizer()
    jobToFindSimVt = tfidf_vectorizer2.fit_transform(jobToFindSim['fulltext'])
    otherJobsVt = tfidf_vectorizer2.transform(jobdf['fulltext'])
    jobSimMatrix = map(lambda x: cosine_similarity(
        otherJobsVt, x), jobToFindSimVt)
    jobSimList = list(jobSimMatrix)
    print("-----1")
    print(jobSimList)
    simXidxList = {str(idx):x[0] for idx, x in enumerate(jobSimList[0])}
    print("-----2")
    print(simXidxList)
    sortedSimXidxList = sorted(simXidxList.items(), key=lambda x: x[1])
    print("-----3")
    print(sortedSimXidxList)
    topSimJobtoJon = sortedSimXidxList[-6:]
    print("-----4")
    print(topSimJobtoJon)
    idxSimList = [int(x[0]) for x in topSimJobtoJon]
    jobSimjobIds = []
    for idx in idxSimList:
        jobSimjobIds.append(jobdf.iloc[idx]['_id'])
    print("-----5")    
    print(jobSimjobIds)    
    jobSimjobIds.pop()    
    return JsonResponse(status=200, data={"sugList": jobSimjobIds})
