#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np 
from NEWSREC import *
import os, pickle
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

data_path = '../tcdata/'
save_path = '../user_data/'
pred_path ='../prediction_result/'

train_name='train_click_log.csv'
test_name='testB_click_log.csv'
article_name='articles.csv'
user_info_name='user_info.csv'
#如果A榜，则改为test_name='testA_click_log.csv'

max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))

all_click_df = get_all_click_df(data_path, offline=False,train_name=train_name, test_name=test_name)
all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)

item_info_df = get_item_info_df(data_path)

if os.path.exists(save_path + 'item_content_emb.pkl'):
    item_emb_dict=pickle.load(open(save_path + 'item_content_emb.pkl', 'rb'))
else:
    item_emb_dict = get_item_emb_dict(data_path,save_path)

item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)

if test_name=='testB_click_log.csv' and os.path.exists("../user_data/datastore/trn_hist_click_df.csv") and os.path.exists("../user_data/datastore/trn_last_click_df.csv"):
    trn_hist_click_df=pd.read_csv("../user_data/datastore/trn_hist_click_df.csv")
    trn_last_click_df=pd.read_csv("../user_data/datastore/trn_last_click_df.csv")
else:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    trn_hist_click_df.to_csv("../user_data/datastore/trn_hist_click_df.csv",index=False)
    trn_last_click_df.to_csv("../user_data/datastore/trn_last_click_df.csv",index=False)

if test_name=='testB_click_log.csv' and os.path.exists(save_path + 'itemcf_i2i_sim_7_5_6_havetest.pkl'):
    i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim_7_5_6_havetest.pkl', 'rb'))
else:
    i2i_sim = itemcf_sim(save_path, all_click_df, item_created_time_dict, offline=False)

item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')
if os.path.exists(save_path + 'emb_i2i_sim.pkl'):
    emb_i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))
else:
    emb_i2i_sim = embdding_sim(all_click_df, item_emb_df, save_path, topk=10) 

if test_name=='testB_click_log.csv' and os.path.exists(save_path + 'itemcf_recall_dict_756_001_3_250_topk50_num10_havetest.pkl'):
    pass
else:
    get_recall(save_path, all_click_df, item_created_time_dict, metric_recall=False)

all_click_df = get_all_click_df(data_path, offline=True, train_name=train_name, test_name=test_name)
all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)

if os.path.exists("../user_data/datastore/trn_hist_click_df_notest1-11-1.csv") and os.path.exists("../user_data/datastore/trn_last_click_df_notest1-11-1.csv"):
    trn_hist_click_df=pd.read_csv("../user_data/datastore/trn_hist_click_df_notest1-11-1.csv")
    trn_last_click_df=pd.read_csv("../user_data/datastore/trn_last_click_df_notest1-11-1.csv")
else:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    trn_hist_click_df.to_csv("../user_data/datastore/trn_hist_click_df_notest1-11-1.csv",index=False)
    trn_last_click_df.to_csv("../user_data/datastore/trn_last_click_df_notest1-11-1.csv",index=False)

if os.path.exists(save_path + 'itemcf_i2i_sim_7_5_6_notest.pkl'):
    i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim_7_5_6_notest.pkl', 'rb'))
else:
    i2i_sim = itemcf_sim(save_path, all_click_df, item_created_time_dict, offline=True)

if os.path.exists(save_path + 'itemcf_recall_dict_756_001_3_250_topk50_num10_notest.pkl'):
    pass
else:
    get_recall(save_path, all_click_df, item_created_time_dict, metric_recall=True)

click_trn, click_val, click_tst, val_ans = get_trn_val_tst_data(data_path,train_name,test_name,validation_set=False)

#click_trn_hist, click_trn_last = get_hist_and_last_click(click_trn)
click_trn_hist=pd.read_csv("../user_data/datastore/trn_hist_click_df_notest1-11-1.csv")
click_trn_last=pd.read_csv("../user_data/datastore/trn_last_click_df_notest1-11-1.csv")
if click_val is not None:
    click_val_hist, click_val_last = click_val, val_ans
else:
    click_val_hist, click_val_last = None, None    
click_tst_hist = click_tst

recall_list_dict =pickle.load(open(save_path + 'itemcf_recall_dict_756_001_3_250_topk50_num10_notest.pkl', 'rb'))
recall_list_dict2=pickle.load(open(save_path + 'itemcf_recall_dict_756_001_3_250_topk50_num10_havetest.pkl', 'rb'))
recall_list_df = recall_dict_2_df(recall_list_dict)
recall_list_df2= recall_dict_2_df(recall_list_dict2)

trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df = get_user_recall_item_label_df(click_trn_hist, 
                                                                                                       click_val_hist, 
                                                                                                       click_tst_hist,
                                                                                                       click_trn_last, 
                                                                                                       click_val_last, 
                                                                                                       recall_list_df,
                                                                                                       recall_list_df2,
                                                                                                       click_val
                                                                                                      )

article_info_df = get_article_info_df(data_path,article_name=article_name)
all_click = click_trn.append(click_tst)
item_content_emb_dict, item_w2v_emb_dict= get_embedding(save_path, all_click)

if os.path.exists(save_path + 'trn_user_item_feats_df.csv'):
    pass
else:
    trn_user_item_label_tuples = trn_user_item_label_df.groupby('user_id').apply(make_tuple_func).reset_index()
    trn_user_item_label_tuples_dict = dict(zip(trn_user_item_label_tuples['user_id'], trn_user_item_label_tuples[0]))
    if val_user_item_label_df is not None:
        val_user_item_label_tuples = val_user_item_label_df.groupby('user_id').apply(make_tuple_func).reset_index()
        val_user_item_label_tuples_dict = dict(zip(val_user_item_label_tuples['user_id'], val_user_item_label_tuples[0]))
    else:
        val_user_item_label_tuples_dict = None 
    trn_user_item_feats_df = create_feature(trn_user_item_label_tuples_dict.keys(), trn_user_item_label_tuples_dict,                                                 click_trn_hist, article_info_df, item_content_emb_dict)
    if val_user_item_label_tuples_dict is not None:
        val_user_item_feats_df = create_feature(val_user_item_label_tuples_dict.keys(), val_user_item_label_tuples_dict,                                                     click_val_hist, article_info_df, item_content_emb_dict)
    else:
        val_user_item_feats_df = None
    trn_user_item_feats_df.to_csv(save_path + 'trn_user_item_feats_df.csv', index=False)
    if val_user_item_feats_df is not None:
        val_user_item_feats_df.to_csv(save_path + 'val_user_item_feats_df.csv', index=False)

if test_name=='testB_click_log.csv' and os.path.exists(save_path + 'tst_user_item_feats_df.csv'):
    pass
else:
    tst_user_item_label_tuples = tst_user_item_label_df.groupby('user_id').apply(make_tuple_func).reset_index()
    tst_user_item_label_tuples_dict = dict(zip(tst_user_item_label_tuples['user_id'], tst_user_item_label_tuples[0]))
    tst_user_item_feats_df = create_feature(tst_user_item_label_tuples_dict.keys(), tst_user_item_label_tuples_dict,                                                 click_tst_hist, article_info_df, item_content_emb_dict)
    tst_user_item_feats_df.to_csv(save_path + 'tst_user_item_feats_df.csv', index=False) 

if test_name=='testB_click_log.csv' and os.path.exists(save_path + 'user_info.csv'):
    pass
else:
    articles =  pd.read_csv(data_path+article_name)
    articles = reduce_mem(articles)
    all_data = click_trn.append(click_tst)
    if click_val is not None:
        all_data = all_data.append(click_val)
    all_data = reduce_mem(all_data)
    all_data = all_data.merge(articles, left_on='click_article_id', right_on='article_id') 
    user_act_fea = active_level(all_data, ['user_id', 'click_article_id', 'click_timestamp'])
    article_hot_fea = hot_level(all_data, ['user_id', 'click_article_id', 'click_timestamp'])    
    device_cols = ['user_id', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type']
    user_device_info = device_fea(all_data, device_cols)
    user_time_hob_cols = ['user_id', 'click_timestamp', 'created_at_ts']
    user_time_hob_info = user_time_hob_fea(all_data, user_time_hob_cols)
    user_category_hob_cols = ['user_id', 'category_id']
    user_cat_hob_info = user_cat_hob_fea(all_data, user_category_hob_cols)
    user_wcou_info = all_data.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_wcou_info.rename(columns={'words_count': 'words_hbo'}, inplace=True)
    user_info = pd.merge(user_act_fea, user_device_info, on='user_id')
    user_info = user_info.merge(user_time_hob_info, on='user_id')
    user_info = user_info.merge(user_cat_hob_info, on='user_id')
    user_info = user_info.merge(user_wcou_info, on='user_id')
    user_info.to_csv(save_path + user_info_name, index=False)   

if test_name=='testB_click_log.csv' and os.path.exists(save_path + 'trn_user_item_feats_df_finalversion.csv') and os.path.exists(save_path + 'tst_user_item_feats_df_finalversion.csv'):
    pass
else:
    user_info = pd.read_csv(save_path + user_info_name)
    if os.path.exists(save_path + 'trn_user_item_feats_df.csv'):
        trn_user_item_feats_df = pd.read_csv(save_path + 'trn_user_item_feats_df.csv')
    if os.path.exists(save_path + 'tst_user_item_feats_df.csv'):
        tst_user_item_feats_df = pd.read_csv(save_path + 'tst_user_item_feats_df.csv')
    if os.path.exists(save_path + 'val_user_item_feats_df.csv'):
        val_user_item_feats_df = pd.read_csv(save_path + 'val_user_item_feats_df.csv')
    else:
        val_user_item_feats_df = None   
    trn_user_item_feats_df = trn_user_item_feats_df.merge(user_info, on='user_id', how='left')
    if val_user_item_feats_df is not None:
        val_user_item_feats_df = val_user_item_feats_df.merge(user_info, on='user_id', how='left')
    else:
        val_user_item_feats_df = None   
    tst_user_item_feats_df = tst_user_item_feats_df.merge(user_info, on='user_id',how='left') 
    articles =  pd.read_csv(data_path+article_name)
    articles = reduce_mem(articles)
    trn_user_item_feats_df = trn_user_item_feats_df.merge(articles, left_on='click_article_id', right_on='article_id')
    if val_user_item_feats_df is not None:
        val_user_item_feats_df = val_user_item_feats_df.merge(articles, left_on='click_article_id', right_on='article_id')
    else:
        val_user_item_feats_df = None
    tst_user_item_feats_df = tst_user_item_feats_df.merge(articles, left_on='click_article_id', right_on='article_id') 
    trn_user_item_feats_df['is_cat_hab'] = trn_user_item_feats_df.apply(lambda x: 1 if x.category_id in set(x.cate_list) else 0, axis=1)
    if val_user_item_feats_df is not None:
        val_user_item_feats_df['is_cat_hab'] = val_user_item_feats_df.apply(lambda x: 1 if x.category_id in set(x.cate_list) else 0, axis=1)
    else:
        val_user_item_feats_df = None
    tst_user_item_feats_df['is_cat_hab'] = tst_user_item_feats_df.apply(lambda x: 1 if x.category_id in set(x.cate_list) else 0, axis=1)    
    del trn_user_item_feats_df['cate_list']
    if val_user_item_feats_df is not None:
        del val_user_item_feats_df['cate_list']
    else:
        val_user_item_feats_df = None    
    del tst_user_item_feats_df['cate_list']
    del trn_user_item_feats_df['article_id']
    if val_user_item_feats_df is not None:
        del val_user_item_feats_df['article_id']
    else:
        val_user_item_feats_df = None  
    del tst_user_item_feats_df['article_id']    
    trn_user_item_feats_df.to_csv(save_path + 'trn_user_item_feats_df_finalversion.csv', index=False)
    if val_user_item_feats_df is not None:
        val_user_item_feats_df.to_csv(save_path + 'val_user_item_feats_df_finalversion.csv', index=False)
    tst_user_item_feats_df.to_csv(save_path + 'tst_user_item_feats_df_finalversion.csv', index=False)    

have_validation = False
trn_user_item_feats_df = pd.read_csv(save_path + 'trn_user_item_feats_df_finalversion.csv')
trn_user_item_feats_df['click_article_id'] = trn_user_item_feats_df['click_article_id'].astype(int)
if have_validation:
    val_user_item_feats_df = pd.read_csv(save_path + 'val_user_item_feats_df_finalversion.csv')
    val_user_item_feats_df['click_article_id'] = val_user_item_feats_df['click_article_id'].astype(int)
else:
    val_user_item_feats_df = None
tst_user_item_feats_df = pd.read_csv(save_path + 'tst_user_item_feats_df_finalversion.csv')
tst_user_item_feats_df['click_article_id'] = tst_user_item_feats_df['click_article_id'].astype(int)
del tst_user_item_feats_df['label']

trn_user_item_feats_df_rank_model = trn_user_item_feats_df.copy()
if have_validation:
    val_user_item_feats_df_rank_model = val_user_item_feats_df.copy() 
tst_user_item_feats_df_rank_model = tst_user_item_feats_df.copy()

lgb_cols = ['sim0', 'time_diff0', 'word_diff0','sim_max', 'sim_min', 'sim_sum', 
            'sim_mean', 'score','click_size', 'time_diff_mean', 'active_level',
            'click_environment','click_deviceGroup', 'click_os', 'click_country', 
            'click_region','click_referrer_type', 'user_time_hob1', 'user_time_hob2',
            'words_hbo', 'category_id', 'created_at_ts','words_count']

trn_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
g_train = trn_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values
if have_validation:
    val_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
    g_val = val_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

def get_kfold_users(trn_df, n=5):
    user_ids = trn_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set
k_fold = 5
trn_df = trn_user_item_feats_df_rank_model
user_set = get_kfold_users(trn_df, n=k_fold)
score_list = []
score_df = trn_df[['user_id', 'click_article_id','label']]
sub_preds = np.zeros(tst_user_item_feats_df_rank_model.shape[0])
for n_fold, valid_user in enumerate(user_set):
    train_idx = trn_df[~trn_df['user_id'].isin(valid_user)] 
    valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]
    train_idx.sort_values(by=['user_id'], inplace=True)
    g_train = train_idx.groupby(['user_id'], as_index=False).count()["label"].values
    valid_idx.sort_values(by=['user_id'], inplace=True)
    g_val = valid_idx.groupby(['user_id'], as_index=False).count()["label"].values
    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                            max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                            learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs= 16)  
    lgb_ranker.fit(train_idx[lgb_cols], train_idx['label'], group=g_train,
                   eval_set=[(valid_idx[lgb_cols], valid_idx['label'])], eval_group= [g_val], 
                   eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ], early_stopping_rounds=50, )
    valid_idx['pred_score'] = lgb_ranker.predict(valid_idx[lgb_cols], num_iteration=lgb_ranker.best_iteration_)
    valid_idx['pred_score'] = valid_idx[['pred_score']].transform(lambda x: norm_sim(x))  
    valid_idx.sort_values(by=['user_id', 'pred_score'])
    valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
    score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])
    if not have_validation:
        sub_preds += lgb_ranker.predict(tst_user_item_feats_df_rank_model[lgb_cols], lgb_ranker.best_iteration_)
score_df_ = pd.concat(score_list, axis=0)
score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(save_path + 'trn_lgb_ranker_feats.csv', index=False)
tst_user_item_feats_df_rank_model['pred_score'] = sub_preds / k_fold
tst_user_item_feats_df_rank_model['pred_score'] = tst_user_item_feats_df_rank_model['pred_score'].transform(lambda x: norm_sim(x))
tst_user_item_feats_df_rank_model.sort_values(by=['user_id', 'pred_score'])
tst_user_item_feats_df_rank_model['pred_rank'] = tst_user_item_feats_df_rank_model.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(save_path + 'tst_lgb_ranker_feats.csv', index=False)

def get_kfold_users(trn_df, n=5):
    user_ids = trn_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set
k_fold = 5
trn_df = trn_user_item_feats_df_rank_model
user_set = get_kfold_users(trn_df, n=k_fold)
score_list = []
score_df = trn_df[['user_id', 'click_article_id', 'label']]
sub_preds = np.zeros(tst_user_item_feats_df_rank_model.shape[0])
for n_fold, valid_user in enumerate(user_set):
    train_idx = trn_df[~trn_df['user_id'].isin(valid_user)] 
    valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]
    lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                            max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                            learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs= 16, verbose=10)  
    lgb_Classfication.fit(train_idx[lgb_cols], train_idx['label'],eval_set=[(valid_idx[lgb_cols], valid_idx['label'])], 
                          eval_metric=['auc', ],early_stopping_rounds=50, )
    valid_idx['pred_score'] = lgb_Classfication.predict_proba(valid_idx[lgb_cols], 
                                                              num_iteration=lgb_Classfication.best_iteration_)[:,1]
    valid_idx.sort_values(by=['user_id', 'pred_score'])
    valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
    score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])
    if not have_validation:
        sub_preds += lgb_Classfication.predict_proba(tst_user_item_feats_df_rank_model[lgb_cols], 
                                                     num_iteration=lgb_Classfication.best_iteration_)[:,1]
score_df_ = pd.concat(score_list, axis=0)
score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(save_path + 'trn_lgb_cls_feats.csv', index=False)
tst_user_item_feats_df_rank_model['pred_score'] = sub_preds / k_fold
tst_user_item_feats_df_rank_model['pred_score'] = tst_user_item_feats_df_rank_model['pred_score'].transform(lambda x: norm_sim(x))
tst_user_item_feats_df_rank_model.sort_values(by=['user_id', 'pred_score'])
tst_user_item_feats_df_rank_model['pred_rank'] = tst_user_item_feats_df_rank_model.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(save_path + 'tst_lgb_cls_feats.csv', index=False)

trn_lgb_ranker_feats = pd.read_csv(save_path + 'trn_lgb_ranker_feats.csv')
trn_lgb_cls_feats = pd.read_csv(save_path + 'trn_lgb_cls_feats.csv')
tst_lgb_ranker_feats = pd.read_csv(save_path + 'tst_lgb_ranker_feats.csv')
tst_lgb_cls_feats = pd.read_csv(save_path + 'tst_lgb_cls_feats.csv')

finall_trn_ranker_feats = trn_lgb_ranker_feats[['user_id', 'click_article_id', 'label']]
finall_tst_ranker_feats = tst_lgb_ranker_feats[['user_id', 'click_article_id']]
for idx, trn_model in enumerate([trn_lgb_ranker_feats, trn_lgb_cls_feats]):
    for feat in [ 'pred_score', 'pred_rank']:
        col_name = feat + '_' + str(idx)
        finall_trn_ranker_feats[col_name] = trn_model[feat]
for idx, tst_model in enumerate([tst_lgb_ranker_feats, tst_lgb_cls_feats]):
    for feat in [ 'pred_score', 'pred_rank']:
        col_name = feat + '_' + str(idx)
        finall_tst_ranker_feats[col_name] = tst_model[feat]

feat_cols = ['pred_score_0', 'pred_rank_0', 'pred_score_1', 'pred_rank_1']
trn_x = finall_trn_ranker_feats[feat_cols]
trn_y = finall_trn_ranker_feats['label']
tst_x = finall_tst_ranker_feats[feat_cols]
lr = LogisticRegression()
lr.fit(trn_x, trn_y)
finall_tst_ranker_feats['pred_score'] = lr.predict_proba(tst_x)[:, 1]

rank_results = finall_tst_ranker_feats[['user_id', 'click_article_id', 'pred_score']]
submit(pred_path, rank_results, topk=5)

