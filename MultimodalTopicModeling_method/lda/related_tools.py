"""
LDA に関する後処理等をまとめた関数群
"""
import numpy as np 
import pandas as pd
import itertools
from sklearn.metrics import f1_score

def swap_(d1, n1, n2):
    d2 = d1.copy()
    d2[d1 == n1] = n2
    d2[d1 == n2] = n1

    return d2

def calc_sim(d1, d2):
    return np.mean(d1 == d2)

def swap_as_max_sim(d1, d2, li_num = None):
    if li_num is None:
        li_num = d2.unique().tolist()
    max_sim = calc_sim(d1, d2)
    d2_out = d2
    for a, b in itertools.combinations(li_num, 2):
        d2_ = swap_(d2, a, b)
        tmp_sim = calc_sim(d1, d2_)
        if tmp_sim > max_sim:
            max_sim = tmp_sim
            d2_out = d2_

    if d2.equals(d2_out):
        return max_sim, d2_out
    else:
        return swap_as_max_sim(d1, d2_out, li_num) 

def swap_topic_no(df: pd.DataFrame, item_grand=None) -> pd.DataFrame:
    """
    swap topics
    """
    df_swap = df.copy()

    if item_grand is None:
        item_grand = df.iloc[:, 0]
    for cname, item in df.iteritems():
        for srce, dest in itertools.combinations(np.unique(df.values), 2):
            # swap
            item_sw = item.copy()
            item_sw[item == srce] = dest
            item_sw[item == dest] = srce
        
            # calc f1 
            f1_before = f1_score(item_grand, item, average='macro')
            f1_after =  f1_score(item_grand, item_sw, average='macro')
            if f1_after > f1_before:
                item = item_sw
            
        df_swap[cname] = item
    
    return df_swap

def summarise_topic_info(df):
    df_topic_matome = df.stack().reset_index().rename(
        columns={'level_0': 'disease', 'level_1':'sample_name', 0:'topic'}
    ).assign(
        method = lambda df: df.sample_name.str.split('_', expand=True).loc[:, 0]
    ).groupby(['disease', 'method'], as_index=True).apply(
        lambda df: df['topic'].value_counts(normalize=True)
    ).reset_index().groupby(['disease', 'method'], as_index=False).apply(
        lambda df: df.query('topic == topic.max()')
    ).rename(columns={'level_2': 'topic', 'topic': 'ratio'}).groupby(['disease', 'method']).agg({
        'topic': lambda x: ' or '.join(x.map(lambda x: str(x))),
        'ratio': lambda x: x.unique()
    }).unstack().reset_index()

    return df_topic_matome