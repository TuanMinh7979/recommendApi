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
tfidf_vectorizer = TfidfVectorizer()
module_dir = os.path.dirname(__file__)  # get current directory
cvFilePath = os.path.join(module_dir, '..\\media\\pcvsdata.xlsx',)
jobFilePath = os.path.join(module_dir, '..\\media\\pjobsdata.xlsx',)


def getSugCvForJob(request, JobId):

   
    cvdf = pd.read_excel(cvFilePath)
    jobdf = pd.read_excel(jobFilePath)
    
    givenJobRow= jobdf.loc[jobdf['_id'] == JobId]
   
    print(givenJobRow)
    ls = []
    ls.append(givenJobRow['fulltext3'].tolist()[0])
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


def getSugJobForCv(request, CvId):

    cvdf = pd.read_excel(cvFilePath)
    jobdf = pd.read_excel(jobFilePath)

    givenCvRow= cvdf.loc[cvdf['_id'] == CvId]
   
    print(givenCvRow)

    ls = []
    ls.append(givenCvRow['fulltext3'].tolist()[0])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

    tfidf_cvs = tfidf_vectorizer.fit_transform(ls)
    tfidf_jobs = tfidf_vectorizer.transform(jobdf['fulltext3'])
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
