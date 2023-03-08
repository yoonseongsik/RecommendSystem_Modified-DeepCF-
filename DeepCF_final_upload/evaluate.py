'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: hexiangnan
'''
import pandas as pd
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
import torch
from utils import TestDataset

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None


def evaluate_model(model, testRatings, testNegatives, K, num_thread,  item_si_data, user_si_data):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    item_si_data = item_si_data
    user_si_data = user_si_data

    hits, ndcgs = [], []
    if num_thread > 1:  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx, item_si_data, user_si_data)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)


def eval_one_rating(idx,  item_si_data, user_si_data):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    
    item_si_data = item_si_data
    user_si_data = user_si_data
    
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    
    df_item_ev = pd.DataFrame(items)
    df_users_ev = pd.DataFrame(users)

    item_si_df = pd.merge(df_item_ev, item_si_data, how = 'left',left_on = 0, right_on='ISBN')
    item_si_embedding = item_si_df.iloc[:, 3:].to_numpy()
    
    user_si_df = pd.merge(df_users_ev, user_si_data, how = 'left',left_on = 0, right_on='User-ID')
    user_si_embedding = user_si_df.iloc[:, 3:].to_numpy()
    
    
    isCuda = torch.cuda.is_available()
    dst = TestDataset(users, items, item_si_embedding, user_si_embedding)
    ldr = torch.utils.data.DataLoader(dst, batch_size=100, shuffle=False)

    _model.eval()
    predictions = [None] * len(dst)
    total = 0
    with torch.no_grad():
        for ui, ii, iem, uem in ldr:
            if isCuda:
                ui, ii, iem, uem = ui.cuda(), ii.cuda(), iem.cuda(), uem.cuda()
            bsz = ui.size(0)
            ri = _model(ui, ii, iem, uem).squeeze().cpu().tolist()
            predictions[total:total+bsz] = ri
    # predictions = _model.predict([users, np.array(items)], 
    #                              batch_size=100, verbose=0)
    # ---
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
