import re
import numpy as np
from string import punctuation
from pymongo import MongoClient
import pymongo
from email import message
from django.shortcuts import render, redirect
from django.http import HttpResponse
# Create your views here.
from django.contrib.auth.models import User, auth
from django.http import JsonResponse
from django.core import serializers

from django.forms.models import model_to_dict
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyvi import ViTokenizer

module_dir = os.path.dirname(__file__)  # get current directory
cvFilePath = os.path.join(module_dir, '../media/cvsDbData.xlsx',)
jobFilePath = os.path.join(module_dir, '../media/jobsDbData.xlsx',)


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
    with open("D:/sugDjango/recommendApi/media/viStop.txt", encoding="utf-8") as f:
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


def readMongoDBData(request):
    client = MongoClient()
    # point the client at mongo URI
    client = MongoClient(
        'mongodb+srv://tuan:12345678Abc@cluster0.h8bya9k.mongodb.net/')
    # #select database
    db = client['jobapp']
    # #select the collection within the database
    jobposts = db.jobposts
    jobpostsDb = pd.DataFrame(list(jobposts.find()))
    resume = db.resumes
    resumeDb = pd.DataFrame(list(resume.find()))

    jobpostsDb = jobpostsDb.fillna('')
    resumeDb = resumeDb.fillna('')
    jobpostsDb['fulltext'] = jobpostsDb['candidateRequiredText'] + \
        jobpostsDb['descriptionText']+jobpostsDb['title']
    resumeDb['fulltext'] = resumeDb['experience']+resumeDb['skills']

    #
    jobs = jobpostsDb[["_id", "fulltext"]]
    cvs = resumeDb[["_id", "fulltext"]]
    # text preprocessing, remove html, punctuation
    jobs['fulltext'] = np.vectorize(cleanhtml)(jobs['fulltext'])
    jobs['fulltext'] = np.vectorize(text_preprocessing)(jobs['fulltext'])
    cvs['fulltext'] = np.vectorize(cleanhtml)(cvs['fulltext'])
    cvs['fulltext'] = np.vectorize(text_preprocessing)(cvs['fulltext'])
    #tokenize
    jobs["fulltext"]=jobs["fulltext"].apply(lambda x: tokenize(x))
    cvs["fulltext"]=cvs["fulltext"].apply(lambda x: tokenize(x))
    # remove stopword
    stw = getStopWord()
    jobs["fulltext"]=jobs["fulltext"].apply(lambda x: rmStw(x, stw))
    cvs["fulltext"]=cvs["fulltext"].apply(lambda x: rmStw(x, stw))

    print(jobs.head())
    print(cvs.head())
    jobs.to_excel(r"D:/sugDjango/recommendApi/media/jobDbData1.xlsx", index=True)
    cvs.to_excel(r"D:/sugDjango/recommendApi/media/cvsDbData1.xlsx", index=True)

    return JsonResponse(status=200, data={"ok": "ok"})


def getSugCvForJob(request, jobId):

    cvdf = pd.read_excel(cvFilePath)
    jobdf = pd.read_excel(jobFilePath)

    givenJobRow = jobdf.loc[jobdf['_id'] == jobId]

    print(givenJobRow)
    ls = []
    ls.append(givenJobRow['fulltext3'].tolist()[0])
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_jobs = tfidf_vectorizer.fit_transform(ls)

    tfidf_cvs = tfidf_vectorizer.transform(cvdf['fulltext3'])

    cos_similarity = map(lambda x: cosine_similarity(tfidf_cvs, x), tfidf_jobs)
    simrs = list(cos_similarity)

    rs = simrs[0]
    kq = {x[0]: idx for idx, x in enumerate(rs)}
    print(kq)
    srs = sorted(kq.items(), key=lambda x: x[0])
    fiveBestSimilar = srs[-5:]
    print(fiveBestSimilar)
    finalRs = [x[1] for x in fiveBestSimilar]
    ids = []
    for idx in finalRs:
        ids.append(cvdf.iloc[idx]['_id'])
    print(ids)

    return JsonResponse(status=200, data={"sugList": ids})


def getSugJobForCv(request, cvId):

    cvdf = pd.read_excel(cvFilePath)
    jobdf = pd.read_excel(jobFilePath)

    givenCvRow = cvdf.loc[cvdf['_id'] == cvId]

    print(givenCvRow)

    ls = []
    ls.append(givenCvRow['fulltext3'].tolist()[0])

    tfidf_vectorizer1 = TfidfVectorizer()
    tfidf_cvs = tfidf_vectorizer1.fit_transform(ls)
    tfidf_jobs = tfidf_vectorizer1.transform(jobdf['fulltext3'])
    cos_similarity = map(lambda x: cosine_similarity(tfidf_jobs, x), tfidf_cvs)
    simrs = list(cos_similarity)

    rs = simrs[0]
    print(rs)
    kq = {x[0]: idx for idx, x in enumerate(rs)}
    # print(kq)
    srs = sorted(kq.items(), key=lambda x: x[0])
    fiveBestSimilar = srs[-5:]

    print(fiveBestSimilar)
    finalRs = [x[1] for x in fiveBestSimilar]
    ids = []
    for idx in finalRs:
        ids.append(jobdf.iloc[idx]['_id'])
    print(ids)
    return JsonResponse(status=200, data={"sugList": ids})


def getSimilarJob(request, jobId):
    jobdf = pd.read_excel(jobFilePath)

    jobToFindSim = jobdf.loc[jobdf['_id'] == jobId]
    tfidf_vectorizer2 = TfidfVectorizer()
    jobToFindSimVt = tfidf_vectorizer2.fit_transform(jobToFindSim['fulltext3'])
    otherJobsVt = tfidf_vectorizer2.transform(jobdf['fulltext3'])
    jobSimMatrix = map(lambda x: cosine_similarity(
        otherJobsVt, x), jobToFindSimVt)
    jobSimList = list(jobSimMatrix)
    simXidxList = {x[0]: idx for idx, x in enumerate(jobSimList[0])}
    sortedSimXidxList = sorted(simXidxList.items(), key=lambda x: x[0])
    topSimJobtoJon = sortedSimXidxList[-6:]
    idxSimList = [x[1] for x in topSimJobtoJon]
    jobSimjobIds = []
    for idx in idxSimList:
        jobSimjobIds.append(jobdf.iloc[idx]['_id'])
    return JsonResponse(status=200, data={"sugList": jobSimjobIds})
