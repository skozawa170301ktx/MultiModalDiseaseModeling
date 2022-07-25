from logging import disable
from math import isnan
from operator import imod
from os import link
import sys
from copy import copy
from collections import Counter, deque
from typing import List, Tuple
from typing import Dict
import numpy as np
from numpy.core.numeric import zeros_like
from numpy.core.numerictypes import maximum_sctype
import itertools
from multiprocessing import Pool
from pandas.core.frame import DataFrame
# from scipy.special import loggamma
from scipy.special import logsumexp
from scipy.stats.mstats import gmean
import pandas as pd
from gensim.corpora import Dictionary
from lda.lda_prop4 import MyCorpus, MyCorpora, Lda

class MultiLDA(Lda):
    def __init__(self, N_topic):
        self.N_topic = N_topic
        
    def sample_w(self, z: List[List], phi: np.array) -> List[np.array]:
        _, n_voc = phi.shape
        rng = np.random.default_rng()
        # 1. 各トピック別に Phi_k に従い単語IDをサンプル
        c = Counter([z_di for z_d in z for z_di in z_d])
        vocs = dict()
        for topic, n_sample in c.items():
            vocs[topic] = list(rng.choice(a=range(n_voc), size=n_sample, p=phi[topic], replace=True))
        
        # 2. 1. でサンプルした単語IDを z_di に従い割り当てる
        #k_index = np.zeros(n_topic, dtype='int') # 各トピック別に何番目の項目まで代入したか
        w = []
        for z_d in z:
            w_d = np.zeros(len(z_d), dtype='int')
            for i, z_di in enumerate(z_d):
                w_d[i] = vocs[z_di].pop()
                #w_d[i] = vocs[z_di][k_index[z_di]]
                #k_index[z_di] += 1
            w.append(w_d)
            
        if self.w_replace is False:
            # 重複を許さない場合の追加処理
            # 重複している個数が3個なら2個だけ再サンプリングしたい
            # pandas のduplicated 関数使いたいのでdataframe にした方が楽では？
            df_wz = pd.DataFrame(
                [
                    [d, w_di, z_di] for d, (w_d, z_d) in enumerate(zip(w, z)) for w_di, z_di in zip(w_d, z_d)
                ], 
                columns=['d', 'w', 'z']
            ).assign(
                dup = lambda df: df.duplicated(subset=['d', 'w'])
            )
            
            for d, group in df_wz.groupby(['d']):
                # ユニークな voc id のリストアップ
                voc_uniq = group.w.unique()
                # 重複している要素の個数とそれぞれに対応するトピック番号のリストアップ
                df_dupd = group.query('dup')
                # 重複していない要素がサンプルされるまで再サンプリング
                for _, row in df_dupd.iterrows():
                    for loop in range(100):
                        cnd_rng = rng.choice(a=range(n_voc), size=1, p=phi[row.z])
                        
                        if cnd_rng not in voc_uniq:
                            voc_uniq = np.append(voc_uniq, cnd_rng)
                            break
                    else:
                        raise ValueError('w_d duplicated')
                    
                if len(w[d]) == len(voc_uniq):
                        w[d] = voc_uniq
                else:
                    print(f'd:{d} w_d: {len(w[d])}, resampled: {len(voc_uniq)}')
                    raise ValueError('resampled corpus length is not same with w_d')

        return w

    def estimate_w(self, corpus, w_index, z, phi) -> List[List[Tuple[int, int]]]:
        '''
        desc:
            観測/未観測を含めた w のコーパスを返り値にする
        input:
            self,
            corpus: 現在の w, 参照渡しになってる,
            w_index: True/False w が未観測か否か
            z:
            phi: トピック数x語彙数の probablistic matrix
        return:
            new corpus
        '''
        if w_index.sum() == 0:
            raise ValueError('input w is fully observed')
        
        target_z = list(itertools.compress(z, w_index)) #z の中で、未観測のものだけを抽出
        # サンプリング
        w_id = self.sample_w(target_z, phi)
        # 一つの文書で同じ単語を2回以上サンプリングした際にちゃんとカウントするための処理
        # スタック・キューの実装は list は不向き（遅いらしい） collection.deque を使用する
        # https://note.nkmk.me/python-collections-deque/
        corpus_notobs = deque(
            [[(d, i) for d, i in dict(Counter(w)).items()] for w in w_id]
        )
        
        corpus_new = copy(corpus) # 元のcopusを破壊しないように
        for i_d, is_not_obs in enumerate(w_index):
            if is_not_obs is True:
                #pass
                corpus_new[i_d] = corpus_notobs.popleft()
        ##
        return corpus_new
    
    def fit(self, n_sample=100, w_replace=True):
        '''
        w に欠損値が無ければ従来の予測を使って、欠損があれば新たに実装した予測方法を使う
        '''
        # ほんとは w_replace だけでなく, n_sample もこの時点でインスタンス変数に格納したいが、
        # 親クラスまで変更しなきゃいけないのでめんどいからパス 
        self.w_replace = w_replace        
        
        is_fullly_observed = self.corpora_disease.corpora_df.isnull().sum().sum() == 0
        if is_fullly_observed:
            super(MultiLDA, self).fit(n_sample)
        else:
            self.fit_disease_estimate_w(n_sample)
    
    def fit_disease_estimate_w(self, n_sample=100):
        '''
        w に欠損がある場合のLDAの制御部分
        '''
        self.n_sample = n_sample

        idx_w_not_obs = self._fit_disease_estimate_w_init()
        
        for n in range(self.n_sample):
            if n % 1 == 0:
                print(f'\r {n}/{self.n_sample}', end='')
            
            # Gibbs sampling body ====================
            self._1step_gibbs(idx_w_not_obs)
            # ========================================
        
        self.theta_disease_log = np.stack(self.theta_disease_log)
        self.phi_disease_log = {key: np.stack(item) for key, item in self.phi_disease_log.items()}

    def _fit_disease_estimate_w_init(self):
        '''
        w の予測：初期値設定
        '''
        # w の not observed 領域の同定
        idx_w_not_obs = self.corpora_disease.corpora_df.isnull()

        self.theta_disease: np.ndarray = np.ones((len(self.corpora_disease.title), self.N_topic))/self.N_topic # document x topic
        self.phi_disease: dict[np.ndarray] = {
            key: np.ones((self.N_topic, len(item.dictionary)))/len(item.dictionary) for key, item in self.corpora_disease.corpora.items()
        } # topic x dictionary
        # ? w の初期値を与える
        # 未観測ポイント
        w_disease_init = self.corpora_disease.init_w()
        
        self.w_disease = dict()
        for key in self.corpora_disease.corpora.keys():
            self.w_disease[key] = MyCorpus(
                corpus = w_disease_init[key].tolist(),
                dictionary = self.corpora_disease.corpora[key].dictionary, 
                title = list(w_disease_init[key].index.values)
            )

        # log 記録
        self.theta_disease_log: List[np.ndarray] = []
        self.phi_disease_log: Dict[str, List[np.ndarray]] = {key: [] for key in self.corpora_disease.corpora.keys()}
        self.w_disease_log: Dict[str, List[pd.DataFrame]] = {key: [] for key in self.corpora_disease.corpora.keys()}

        return idx_w_not_obs

    def _1step_gibbs(self, idx_w_not_obs):
        '''
        wの予測：サンプリングの1ステップ
        '''
        # sampling ============================
        # z
        z_disease = {key: self.sample_z(self.w_disease[key], self.theta_disease, self.phi_disease[key]) for key in self.corpora_disease.corpora.keys()}               
        # theta
        theta_disease_1 = self.sample_theta_drug(z_disease, self.alpha_0_disease)
        # phi
        phi_disease_1 = {key: self.sample_phi(self.w_disease[key], z_disease[key]) for key in self.corpora_disease.corpora.keys()}

        # w_disease
        for key in self.corpora_disease.corpora.keys():
            if idx_w_not_obs[key].sum() > 0:
                new_w = self.estimate_w(self.w_disease[key].corpus, idx_w_not_obs[key], z_disease[key], phi_disease_1[key])
                # kousin
                self.w_disease[key].corpus = new_w
                # self.w_disease[key].update_corpus_df() #あった方がいいけど後の処理に影響がないので省略            
                # # log処理
                df_w_log = pd.DataFrame(
                    [(i, cp) for i, cp in zip(idx_w_not_obs[key].index, new_w) if idx_w_not_obs[key][i]], 
                    columns=['key_c', 'sampled_w']
                )
                self.w_disease_log[key].append(df_w_log)
            else: 
                pass
                # w に欠損値がない場合はｗの推定を省略したらいいだけだからエラー出す必要はない
                # けどデバッグ段階だしとりあえず出すようにしとこ
                # raise ValueError(f'In function: {sys._getframe().f_code.co_name}: there are no lack value in input w at "{key}".')
                
        # record sample ========================
        self.theta_disease_log.append(theta_disease_1)
        for key in self.corpora_disease.corpora.keys():
            self.phi_disease_log[key].append(phi_disease_1[key])

        # 更新 ==================================
        self.theta_disease = theta_disease_1
        self.phi_disease = phi_disease_1


    ################################################################
    # LDA の推定が終わった後の処理
    # functions for post-processing
    ################################################################
    def get_w_estimated(self, burn_in=0):
        ''' Return the predicted values of the missing w
        Parameters
        ---------- 
        burn_in: int 
            Burn-in period for expected value estimation of w.
            
        Returns
        -------
        predicted_w: Dataframe
            dataframe that include expredcted values of w            
        '''
        keys = self.corpora_disease.corpora.keys()
        li_w = []
        for key in keys:
            if len(self.w_disease_log[key]) == 0:
                continue
            
            li_w_key = []
            w_disease_log = self.w_disease_log[key][burn_in:]
            key_dict = self.corpora_disease.corpora[key].dictionary
            for sample_i, df_new_w in enumerate(w_disease_log):
                for ind, row in df_new_w.iterrows():
                    df_tmp = pd.DataFrame(
                        [(w_id, key_dict[w_id], w_count) for w_id, w_count in row['sampled_w']],
                        columns=['word_id', 'word', 'w_count']
                    ).assign(
                        sample_no = sample_i+burn_in,
                        key_c =  row['key_c'],
                        element_name = key,
                    )
                    
                    li_w_key.append(df_tmp)
            
            # make dataframe about expected w at each element-type
            df_w_key = pd.concat(li_w_key).groupby(
                ['element_name', 'key_c', 'word'], as_index=False
            ).agg({
                'w_count': sum,
            }).assign(
                freq = lambda df: df.w_count/len(w_disease_log)
            )
            
            li_w.append(df_w_key)
        
        predicted_w = pd.concat(li_w)
        
        return predicted_w
    
    def get_estimated_topics(self, n_samples_to_use=None):
        '''Return the probability of topics and the most likily topic at each disease.
        Parameters
        ----------
        self
        samples_used: int
            # of samples for calculating the sample means.
            
        Returns
        -------
        df_topic_prop: pd.DataFrame
            diseases x (topic_prop, most_likely_topic). 
        '''
        if not hasattr(self, 'theta_disease_log'):
            return None
        
        if n_samples_to_use is None:
            theta_log = self.theta_disease_log
        else:
            theta_log = self.theta_disease_log[-n_samples_to_use:, :, :]
        
        # get sample mean of topic theta, and mostly like topic number at each disease.
        df_topic_prop = pd.DataFrame(
            np.mean(theta_log, axis=0),
            index=self.corpora_disease.title
        ).assign(
            topic = np.argmax(np.mean(theta_log, axis=0), axis=1), 
        )
        return df_topic_prop
    
    
    def output_score_components(self, topic_dist, top_n = None):
        ''' Output likely score marix associate with the topic_distribution
        Parameter
        ---------
        topic_dist
        top_n
        
        Return
        ------
        dict_out
        '''
        
        if  self.N_topic != len(topic_dist):
            raise ValueError('topic_dise dim not appropriate.')
        
        topic_dist = np.array(topic_dist)
        
        dict_out = dict()
        for key in self.corpora_disease.corpora.keys():
            score_weighted_by_topic_dist = self.phi_disease_log[key].mean(axis=0)*topic_dist[:, np.newaxis]
            
            df_score = pd.DataFrame({
                'component': [self.corpora_disease.corpora[key].dictionary[i] for i in range(score_weighted_by_topic_dist.shape[1])],
                'prob': score_weighted_by_topic_dist.sum(axis=0)                     
            }).sort_values('prob', ascending=False)
            
            if top_n is None:
                dict_out[key] = df_score
            else: 
                dict_out[key] = df_score.iloc[:top_n, :]
            
        return dict_out
            
    def get_prob_components_at_disease(self, key_c_name, n_sample=None):
        ''' Probability of components when input topic ditribution of `key_c_name`
        Parameter
        --------- 
        key_c_name: str
            key_c (Disease) name to desire to obtain the probability of components.
        n_sample: int
            # of samples to get topic ditributions
        
        Return
        ------
        dict_comp: Dict[pd.DataFrame]
            the probabilities of components at each DB
        '''
        # get the topic distribution of `key_c_name`
        df_topic = self.get_estimated_topics(n_samples_to_use=n_sample).loc[key_c_name, :].drop('topic')
        # get component scores for the topic distribution
        dic_score = self.output_score_components(df_topic)
        # ranking probability the original and the other words in the `key_c_name`       
        dic_comp = dict() 
        for key, item in dic_score.items():
            a = pd.merge(
                self.corpora_disease.corpora[key].df_chi().query('document == @key_c_name'), 
                item,
                left_on = 'vocabulary',
                right_on= 'component',
                how='outer'
            ).assign(
                prob_rank = lambda df: df.prob.rank(ascending=False)
            ).sort_values('prob_rank').drop('vocabulary', axis=1)
            dic_comp[key] = a
            
        return dic_comp
    
    def estimate_topics(self, corpora: Dict[str, MyCorpus], n_sample: int=1000):
        ''' 単語群を入力したら推定されるトピックを出力する関数
        複数DB対応、欠損値非対応 (欠損値がある場合は別個にやってね)
        
        '''
        corpora_input = MyCorpora()
        corpora_input.input_new_corpus(corpora)  
        
        theta_disease = np.ones([len(corpora_input.title), self.N_topic])/self.N_topic
        phi_disease = {
            key: np.mean(self.phi_disease_log[key], axis=0)
            for key in corpora_input.corpora.keys()
        }               
        
        theta_disease_1 = []
        for i in range(n_sample):
            # z
            z_disease = {
                key: self.sample_z(corpora_input.corpora[key], theta_disease, phi_disease[key]) 
                for key in corpora_input.corpora.keys()
            }               
            # theta
            theta_disease = self.sample_theta_drug(z_disease, self.alpha_0_disease)
            theta_disease_1.append(theta_disease)
            
        return np.stack(theta_disease_1)

        
        
        
if __name__ == '__main__':
    pass
