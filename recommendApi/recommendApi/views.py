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
    ls = []
    ls.append(jobdf.iloc[0]['jobdata'])
    tfidf_jobs = tfidf_vectorizer.fit_transform(ls)

    tfidf_cvs = tfidf_vectorizer.transform(cvdf['cvdata'])

    cos_similarity = map(lambda x: cosine_similarity(tfidf_cvs, x), tfidf_jobs)
    simrs = list(cos_similarity)

    rs = simrs[0]
    kq = {x[0]: idx for idx, x in enumerate(rs)}
    print(kq)
    srs = sorted(kq.items(), key=lambda x: x[0])
    fiveBestSimilar = srs[-5:]
    finalRs = [x[1] for x in fiveBestSimilar]
    ids = []
    for idx in finalRs:
        ids.append(cvdf.iloc[idx]['_id'])
    print(ids)
    return JsonResponse(status=200, data={"sugList": finalRs})


def getSugJobForCv(request, CvId):

    cvdf = pd.read_excel(cvFilePath)
    jobdf = pd.read_excel(jobFilePath)
    ls = []
    ls.append(cvdf.iloc[0]['cvdata'])
    tfidf_cvs = tfidf_vectorizer.fit_transform(ls)

    tfidf_jobs = tfidf_vectorizer.transform(jobdf['jobdata'])

    cos_similarity = map(lambda x: cosine_similarity(tfidf_jobs, x), tfidf_cvs)
    simrs = list(cos_similarity)

    rs = simrs[0]
    print(rs)
    kq = {x[0]: idx for idx, x in enumerate(rs)}
    # print(kq)
    srs = sorted(kq.items(), key=lambda x: x[0])
    fiveBestSimilar = srs[-5:]
    print("-----")
    print(fiveBestSimilar)
    finalRs = [x[1] for x in fiveBestSimilar]
    ids = []
    for idx in finalRs:
        ids.append(jobdf.iloc[idx]['_id'])
    print(ids)
    return JsonResponse(status=200, data={"sugList": finalRs})
