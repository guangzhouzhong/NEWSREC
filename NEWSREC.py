#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np 
from collections import defaultdict  
import os, math, warnings, math, pickle, random, gc, time,logging
from tqdm import tqdm
import lightgbm as lgb
from gensim.models import Word2Vec
import faiss
import collections
from operator import itemgetter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss
warnings.filterwarnings('ignore')

def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                    100*(start_mem-end_mem)/start_mem,
                                                                                                    (time.time()-starttime)/60))
    return df

def get_all_click_df(data_path='./', offline=True, train_name='train_click_log.csv', test_name='testB_click_log_Test_B.csv'):
    if offline:
        all_click = pd.read_csv(data_path + train_name)
    else:
        trn_click = pd.read_csv(data_path + train_name)
        tst_click = pd.read_csv(data_path + test_name)
        all_click = trn_click.append(tst_click)   
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

def get_item_info_df(data_path):
    item_info_df = pd.read_csv(data_path + 'articles.csv')
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})    
    return item_info_df

def get_item_emb_dict(data_path):
    item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    pickle.dump(item_emb_dict, open(save_path + 'item_content_emb.pkl', 'wb'))   
    return item_emb_dict

def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')    
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))    
    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(lambda x: make_item_time_pair(x))                                                            .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))    
    return user_item_time_dict

def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))    
    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')['user_id', 'click_timestamp'].apply(lambda x: make_user_time_pair(x))                                                            .reset_index().rename(columns={0: 'user_time_list'})    
    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict

def get_hist_and_last_click(all_click):    
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]
    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)
    return click_hist_df, click_last_df

def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)    
    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))    
    return item_type_dict, item_words_dict, item_created_time_dict

def get_user_hist_item_info_dict(all_click):
    user_hist_item_typs = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict = dict(zip(user_hist_item_typs['user_id'], user_hist_item_typs['category_id']))
    user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict = dict(zip(user_hist_item_ids_dict['user_id'], user_hist_item_ids_dict['click_article_id']))
    user_hist_item_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count']))
    all_click_ = all_click.sort_values('click_timestamp')
    user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).reset_index()
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(max_min_scaler)
    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'],                                                 user_last_item_created_time['created_at_ts']))
    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict

def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click

def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=50):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict) 
    for k in range(10, topk+1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1        
        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)

def itemcf_sim(df, item_created_time_dict, offline= False):
    user_item_time_dict = get_user_item_time(df)
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if(i == j):
                    continue
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.7 ** (np.abs(loc2 - loc1) - 1))
                click_time_weight = np.exp(0.5 ** np.abs(i_click_time - j_click_time))
                created_time_weight = np.exp(0.6 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(len(item_time_list) + 1)                
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    if offline:
        pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim_7_5_6_notest.pkl', 'wb'))
    else:
        pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim_7_5_6_havetest.pkl', 'wb')) 
    return i2i_sim_

def embdding_sim(click_df, item_emb_df, save_path, topk):
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))    
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    sim, idx = item_index.search(item_emb_np, topk) 
    item_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]): 
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value
    pickle.dump(item_sim_dict, open(save_path + 'emb_i2i_sim.pkl', 'wb'))       
    return item_sim_dict

def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim):
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}    
    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue
            created_time_weight = np.exp(0.001 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
            loc_weight = (0.3 ** (len(user_hist_items) - loc))            
            content_weight = 25.0
            if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                content_weight += emb_i2i_sim[i][j]
            if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                content_weight += emb_i2i_sim[j][i]            
            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * content_weight * wij
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items(): 
                continue
            item_rank[item] = - i - 100 
            if len(item_rank) == recall_item_num:
                break
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]        
    return item_rank

def get_recall(metric_recall=False):
    if metric_recall:
        trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    else:
        trn_hist_click_df = all_click_df
    user_recall_items_dict = collections.defaultdict(dict)
    user_item_time_dict = get_user_item_time(trn_hist_click_df)
    i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim_7_5_6_havetest.pkl', 'rb'))
    emb_i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))
    sim_item_topk = 50
    recall_item_num = 10
    item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

    for user in tqdm(trn_hist_click_df['user_id'].unique()):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict,                                                             i2i_sim, sim_item_topk, recall_item_num,                                                             item_topk_click, item_created_time_dict, emb_i2i_sim)
    if metric_recall:
        metrics_recall(user_recall_items_dict, trn_last_click_df, topk=recall_item_num)
        pickle.dump(user_recall_items_dict, open(save_path + 'itemcf_recall_dict_756_001_3_250_topk50_num10_notest.pkl', 'wb'))
    else:
        pickle.dump(user_recall_items_dict, open(save_path + 'itemcf_recall_dict_756_001_3_250_topk50_num10_havetest.pkl', 'wb'))

def trn_val_split(all_click_df, sample_user_nums=10000):
    all_click = all_click_df
    all_user_ids = all_click.user_id.unique()
    sample_user_ids = np.random.choice(all_user_ids, size=sample_user_nums, replace=False) 
    click_val = all_click[all_click['user_id'].isin(sample_user_ids)]
    click_trn = all_click[~all_click['user_id'].isin(sample_user_ids)]
    click_val = click_val.sort_values(['user_id', 'click_timestamp'])
    val_ans = click_val.groupby('user_id').tail(1)
    click_val = click_val.groupby('user_id').apply(lambda x: x[:-1]).reset_index(drop=True)
    val_ans = val_ans[val_ans.user_id.isin(click_val.user_id.unique())]
    click_val = click_val[click_val.user_id.isin(val_ans.user_id.unique())]
    return click_trn, click_val, val_ans

def get_trn_val_tst_data(data_path, validation_set=True, train_name='train_click_log.csv', test_name='testB_click_log_Test_B.csv'):
    if validation_set:
        click_trn_data = pd.read_csv(data_path + train_name)  # 训练集用户点击日志
        click_trn_data = reduce_mem(click_trn_data)
        click_trn, click_val, val_ans = trn_val_split(click_trn_data, sample_user_nums=10000)
    else:
        click_trn = pd.read_csv(data_path + train_name)
        click_trn = reduce_mem(click_trn)
        click_val = None
        val_ans = None
    click_tst = pd.read_csv(data_path + test_name)   
    return click_trn, click_val, click_tst, val_ans

def trian_item_word2vec(click_df, embed_size=64, save_name='item_w2v_emb.pkl', split_char=' '):
    click_df = click_df.sort_values('click_timestamp')
    click_df['click_article_id'] = click_df['click_article_id'].astype(str)
    docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
    docs = docs['click_article_id'].values.tolist()
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    w2v = Word2Vec(docs, size=16, sg=1, window=5, seed=2020, workers=24, min_count=1, iter=1)
    item_w2v_emb_dict = {k: w2v[k] for k in click_df['click_article_id']}
    pickle.dump(item_w2v_emb_dict, open(save_path + 'item_w2v_emb.pkl', 'wb'))    
    return item_w2v_emb_dict

def get_embedding(save_path, all_click_df):
    if os.path.exists(save_path + 'item_content_emb.pkl'):
        item_content_emb_dict = pickle.load(open(save_path + 'item_content_emb.pkl', 'rb'))
    if os.path.exists(save_path + 'item_w2v_emb.pkl'):
        item_w2v_emb_dict = pickle.load(open(save_path + 'item_w2v_emb.pkl', 'rb'))
    else:
        item_w2v_emb_dict = trian_item_word2vec(all_click_df)          
    return item_content_emb_dict, item_w2v_emb_dict

def get_article_info_df(article_name='articles.csv'):
    article_info_df = pd.read_csv(data_path + article_name)
    article_info_df = reduce_mem(article_info_df)    
    return article_info_df

def recall_dict_2_df(recall_list_dict):
    df_row_list = [] 
    for user, recall_list in tqdm(recall_list_dict.items()):
        for item, score in recall_list:
            df_row_list.append([user, item, score])
    col_names = ['user_id', 'sim_item', 'score']
    recall_list_df = pd.DataFrame(df_row_list, columns=col_names)    
    return recall_list_df

def neg_sample_recall_data(recall_items_df, sample_rate=0.001):
    pos_data = recall_items_df[recall_items_df['label'] == 1]
    neg_data = recall_items_df[recall_items_df['label'] == 0]   
    print('pos_data_num:', len(pos_data), 'neg_data_num:', len(neg_data), 'pos/neg:', len(pos_data)/len(neg_data))    
    def neg_sample_func(group_df):
        neg_num = len(group_df)
        sample_num = max(int(neg_num * sample_rate), 1)
        sample_num = min(sample_num, 5) 
        return group_df.sample(n=sample_num, replace=True)
    neg_data_user_sample = neg_data.groupby('user_id', group_keys=False).apply(neg_sample_func)
    neg_data_item_sample = neg_data.groupby('sim_item', group_keys=False).apply(neg_sample_func)
    neg_data_new = neg_data_user_sample.append(neg_data_item_sample)
    neg_data_new = neg_data_new.sort_values(['user_id', 'score']).drop_duplicates(['user_id', 'sim_item'], keep='last')
    data_new = pd.concat([pos_data, neg_data_new], ignore_index=True)
    return data_new

def get_rank_label_df(recall_list_df, label_df, is_test=False):
    if is_test:
        recall_list_df['label'] = -1
        return recall_list_df
    label_df = label_df.rename(columns={'click_article_id': 'sim_item'})
    recall_list_df_ = recall_list_df.merge(label_df[['user_id', 'sim_item', 'click_timestamp']],                                                how='left', on=['user_id', 'sim_item'])
    recall_list_df_['label'] = recall_list_df_['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
    del recall_list_df_['click_timestamp']
    return recall_list_df_

def get_user_recall_item_label_df(click_trn_hist, click_val_hist, click_tst_hist,click_trn_last, click_val_last, recall_list_df,recall_list_df2):
    trn_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_trn_hist['user_id'].unique())]
    trn_user_item_label_df = get_rank_label_df(trn_user_items_df, click_trn_last, is_test=False)
    trn_user_item_label_df = neg_sample_recall_data(trn_user_item_label_df)
    if click_val is not None:
        val_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_val_hist['user_id'].unique())]
        val_user_item_label_df = get_rank_label_df(val_user_items_df, click_val_last, is_test=False)
        val_user_item_label_df = neg_sample_recall_data(val_user_item_label_df)
    else:
        val_user_item_label_df = None
    tst_user_items_df = recall_list_df2[recall_list_df2['user_id'].isin(click_tst_hist['user_id'].unique())]
    tst_user_item_label_df = get_rank_label_df(tst_user_items_df, None, is_test=True)
    return trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df

def make_tuple_func(group_df):
    row_data = []
    for name, row_df in group_df.iterrows():
        row_data.append((row_df['sim_item'], row_df['score'], row_df['label']))
    return row_data

def create_feature(users_id, recall_list, click_hist_df,  articles_info, articles_emb, user_emb=None, N=1):
    all_user_feas = []
    i = 0
    for user_id in tqdm(users_id):
        hist_user_items = click_hist_df[click_hist_df['user_id']==user_id]['click_article_id'][-N:]
        for rank, (article_id, score, label) in enumerate(recall_list[user_id]):
            a_create_time = articles_info[articles_info['article_id']==article_id]['created_at_ts'].values[0]
            a_words_count = articles_info[articles_info['article_id']==article_id]['words_count'].values[0]
            single_user_fea = [user_id, article_id]
            sim_fea = []
            time_fea = []
            word_fea = []
            for hist_item in hist_user_items:
                b_create_time = articles_info[articles_info['article_id']==hist_item]['created_at_ts'].values[0]
                b_words_count = articles_info[articles_info['article_id']==hist_item]['words_count'].values[0]
                sim_fea.append(np.dot(articles_emb[hist_item], articles_emb[article_id]))
                time_fea.append(abs(a_create_time-b_create_time))
                word_fea.append(abs(a_words_count-b_words_count))
            single_user_fea.extend(sim_fea)  
            single_user_fea.extend(time_fea)   
            single_user_fea.extend(word_fea)  
            single_user_fea.extend([max(sim_fea), min(sim_fea), sum(sim_fea), sum(sim_fea) / len(sim_fea)])  
            if user_emb:  
                single_user_fea.append(np.dot(user_emb[user_id], articles_emb[article_id]))       
            single_user_fea.extend([score, rank, label])    
            all_user_feas.append(single_user_fea)
    id_cols = ['user_id', 'click_article_id']
    sim_cols = ['sim' + str(i) for i in range(N)]
    time_cols = ['time_diff' + str(i) for i in range(N)]
    word_cols = ['word_diff' + str(i) for i in range(N)]
    sat_cols = ['sim_max', 'sim_min', 'sim_sum', 'sim_mean']
    user_item_sim_cols = ['user_item_sim'] if user_emb else []
    user_score_rank_label = ['score', 'rank', 'label']
    cols = id_cols + sim_cols + time_cols + word_cols + sat_cols + user_item_sim_cols + user_score_rank_label
    df = pd.DataFrame( all_user_feas, columns=cols)
    return df

 def active_level(all_data, cols):
    data = all_data[cols]
    data.sort_values(['user_id', 'click_timestamp'], inplace=True)
    user_act = pd.DataFrame(data.groupby('user_id', as_index=False)[['click_article_id', 'click_timestamp']].                            agg({'click_article_id':np.size, 'click_timestamp': {list}}).values, columns=['user_id', 'click_size', 'click_timestamp'])
    def time_diff_mean(l):
        if len(l) == 1:
            return 1
        else:
            return np.mean([j-i for i, j in list(zip(l[:-1], l[1:]))])
    user_act['time_diff_mean'] = user_act['click_timestamp'].apply(lambda x: time_diff_mean(x))
    user_act['click_size'] = 1 / user_act['click_size']
    user_act['click_size'] = (user_act['click_size'] - user_act['click_size'].min()) / (user_act['click_size'].max() - user_act['click_size'].min())
    user_act['time_diff_mean'] = (user_act['time_diff_mean'] - user_act['time_diff_mean'].min()) / (user_act['time_diff_mean'].max() - user_act['time_diff_mean'].min())     
    user_act['active_level'] = user_act['click_size'] + user_act['time_diff_mean']
    user_act['user_id'] = user_act['user_id'].astype('int')
    del user_act['click_timestamp']
    return user_act

 def hot_level(all_data, cols):
    data = all_data[cols]
    data.sort_values(['click_article_id', 'click_timestamp'], inplace=True)
    article_hot = pd.DataFrame(data.groupby('click_article_id', as_index=False)[['user_id', 'click_timestamp']].                               agg({'user_id':np.size, 'click_timestamp': {list}}).values, columns=['click_article_id', 'user_num', 'click_timestamp'])
    def time_diff_mean(l):
        if len(l) == 1:
            return 1
        else:
            return np.mean([j-i for i, j in list(zip(l[:-1], l[1:]))])
    article_hot['time_diff_mean'] = article_hot['click_timestamp'].apply(lambda x: time_diff_mean(x))
    article_hot['user_num'] = 1 / article_hot['user_num']
    article_hot['user_num'] = (article_hot['user_num'] - article_hot['user_num'].min()) / (article_hot['user_num'].max() - article_hot['user_num'].min())
    article_hot['time_diff_mean'] = (article_hot['time_diff_mean'] - article_hot['time_diff_mean'].min()) / (article_hot['time_diff_mean'].max() - article_hot['time_diff_mean'].min())     
    article_hot['hot_level'] = article_hot['user_num'] + article_hot['time_diff_mean']
    article_hot['click_article_id'] = article_hot['click_article_id'].astype('int')
    del article_hot['click_timestamp']
    return article_hot

def device_fea(all_data, cols):
    user_device_info = all_data[cols]
    user_device_info = user_device_info.groupby('user_id').agg(lambda x: x.value_counts().index[0]).reset_index()
    return user_device_info

def user_time_hob_fea(all_data, cols):
    user_time_hob_info = all_data[cols]
    mm = MinMaxScaler()
    user_time_hob_info['click_timestamp'] = mm.fit_transform(user_time_hob_info[['click_timestamp']])
    user_time_hob_info['created_at_ts'] = mm.fit_transform(user_time_hob_info[['created_at_ts']])
    user_time_hob_info = user_time_hob_info.groupby('user_id').agg('mean').reset_index()
    user_time_hob_info.rename(columns={'click_timestamp': 'user_time_hob1', 'created_at_ts': 'user_time_hob2'}, inplace=True)
    return user_time_hob_info

def user_cat_hob_fea(all_data, cols):
    user_category_hob_info = all_data[cols]
    user_category_hob_info = user_category_hob_info.groupby('user_id').agg({list}).reset_index() 
    user_cat_hob_info = pd.DataFrame()
    user_cat_hob_info['user_id'] = user_category_hob_info['user_id']
    user_cat_hob_info['cate_list'] = user_category_hob_info['category_id']    
    return user_cat_hob_info

def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk
    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()
    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2', 
                                                  3: 'article_3', 4: 'article_4', 5: 'article_5'})
    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)

def norm_sim(sim_df, weight=0.0):
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))
    sim_df = sim_df.apply(lambda sim: sim + weight)
    return sim_df

