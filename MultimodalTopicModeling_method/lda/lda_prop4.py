from logging import disable
from math import isnan
from operator import imod
from os import link
from collections import Counter
from typing import List, Tuple
from typing import Dict
import numpy as np
from numpy.core.numeric import zeros_like
from numpy.core.numerictypes import maximum_sctype
import itertools
from multiprocessing import Pool
from scipy.special import loggamma
from scipy.special import logsumexp
from scipy.stats.mstats import gmean
import pandas as pd
from  gensim.corpora import Dictionary

class MyCorpus:
    def __init__(self, corpus, dictionary, title: List[str]):
        if len(corpus) != len(title):
            raise ValueError("corpus and title must have same length.")

        self.corpus = corpus
        self.dictionary = dictionary
        self.title = title
        self.corpus_df = pd.DataFrame([[d, w_d] for d, w_d in zip(title, corpus)], columns=['title', 'item'])
        
    def df_chi(self):
        corpus_mat = [[d, self.title[d], w_di[0], self.dictionary[w_di[0]], w_di[1]] for d, w_d in enumerate(self.corpus) for w_di in w_d]
        df = pd.DataFrame(
            corpus_mat,
            columns=['document_no', 'document', 'vocabulary_no', 'vocabulary', 'count']
        )
        return df
    
    def fit_param_binom(self) -> Tuple[int, float]:
        """[summary]
        コーパスに含まれている各文書の単語数の分布を推定する(二項分布)

        Returns:
            Tuple[int, float]: [description]
        """        
        N_voc = len(self.dictionary)
        data = self.corpus_df.item.map(len)
        p = data.mean()/N_voc
        
        return N_voc, p
    
    def update_corpus_df(self):
        self.corpus_df = pd.DataFrame([[d, w_d] for d, w_d in zip(self.title, self.corpus)], columns=['title', 'item'])

    def count_total_words(self) -> int:
        poparray = np.array([wdi[1] for wd in self.corpus for wdi in wd])
        
        return poparray.sum()
            
    def get_terms(self, title):
        i = list(self.title).index(title)

        if isinstance(i, int):
            return self.corpus[i]
    
    def make_new_corpus(self, df, columns_title: str, columns_text: str):
        ''' 自身のdictionary を使い回して新しいCorpus を生成
        追加日: 2021/10/28 Satoshi Kozawa
        '''
        title = df[columns_title].tolist()
        texts = df[columns_text].tolist()
        dictionary = self.dictionary
        corpus = [dictionary.doc2bow(doc) for doc in texts]
        
        return MyCorpus(corpus, dictionary, title)
        

    @staticmethod
    def read_corpus_from_dataframe(df, columns_title: str, columns_text: str):
        title = df[columns_title].tolist()
        texts = df[columns_text].tolist()
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(doc) for doc in texts]
        
        return MyCorpus(corpus, dictionary, title)


# 複数の Corpus の集合
# 仕事
#   1. 複数の Corpus の title をそろえること
#   2. 複数の Corpus 間で データの順番が対応できるようにすること
class MyCorpora:
    def __init__(self): 
        self.title = set()
        #self.dictionary = None
        #self.corpus = None
        self.corpora = {}
        # self.corpora_df = pd.DataFrame()

    def input_new_corpus(self, corpora: Dict[str, MyCorpus]):
        # check duplicated elements
        duplicated_elements = set(corpora.keys()) & set(self.corpora.keys())
        if len(duplicated_elements) > 0:
            raise ValueError(f'{duplicated_elements} are already exist.')
        
        # renew attributes
        self.title = self.title.union({title for key, item in corpora.items() for title in item.title})
        self.corpora = corpora
        self.update_corpora_df() 

    def update_corpora_df(self):
        '''
        corpora の状態に併せて corpora_df を更新するメソッド
        '''
        corpora_df = pd.DataFrame()
        li_corpora = []
        for key, item in self.corpora.items():
            hoge = item.corpus_df.assign(elem = key)
            li_corpora.append(hoge)

        corpora_df = pd.concat(li_corpora)
            
        self.corpora_df = corpora_df.set_index(['title', 'elem']).unstack().loc[self.title]
        self.corpora_df.columns=self.corpora_df.columns.droplevel(0)
        
    def init_w(self):
        df_w = pd.DataFrame(index=self.corpora_df.index, columns=self.corpora_df.columns)
        for key, item in self.corpora_df.iteritems():
            item_out = item.copy()
            # このデータセットの単語数分布
            n, p = self.corpora[key].fit_param_binom()
            # 単語数の決定
            N_words = np.random.binomial(n, p, len(item))
            for i, w_d in enumerate(item):
                if not isinstance(w_d, list):
                    words = np.random.choice(range(n), N_words[i], replace=False)
                    item_out[i] = [(w, 1) for w in words]
            
            df_w.loc[:, key] = item_out
        
        return df_w
        
        
class Lda:
    alpha_0_disease = 0.1
    alpha_0_drug = 0.1
    beta_0 = 0.1
    fitted = []

    def __init__(self, N_topic):
        self.N_topic = N_topic

    def load_disease(self, corpora: Dict[str, MyCorpus]) -> None:
        '''
        Disease-xxx data の読み込み
        '''
        self.corpora_disease = MyCorpora()
        self.corpora_disease.input_new_corpus(corpora)  
        
        print(list(self.corpora_disease.corpora.keys()))
        self.theta_disease: np.ndarray = np.ones((len(self.corpora_disease.title), self.N_topic))/self.N_topic # document x topic
        self.phi_disease: dict[np.ndarray] = {
            key: np.ones((self.N_topic, len(item.dictionary)))/len(item.dictionary) for key, item in self.corpora_disease.corpora.items()
        } # topic x dictionary


    #####################################################################################
    # estimation method
    #####################################################################################
    def fit(self, n_sample=100):
        self.n_sample = n_sample
        self.fit_disease()


    # ###########################
    def fit_disease(self):
        # theta の初期値
        self.theta_disease: np.ndarray = np.ones((len(self.corpora_disease.title), self.N_topic))/self.N_topic # document x topic
        # phi の初期値
        self.phi_disease: dict[np.ndarray] = {
            key: np.ones((self.N_topic, len(item.dictionary)))/len(item.dictionary) for key, item in self.corpora_disease.corpora.items()
        } # topic x dictionary
        
        # log 記録
        theta_disease_log: List[np.ndarray] = []
        phi_disease_log: Dict[str, List[np.ndarray]] = {key: [] for key in self.corpora_disease.corpora.keys()}
        # self.z_disease_log: Dict[str, List[List]] = []
        
        # サンプリング開始 ==========================
        for n in range(self.n_sample):
            if n % 50 == 0:
                print(f'\r {n}/{self.n_sample}', end='')
            
            # disease
            ## z
            z_disease = {key: self.sample_z(self.corpora_disease.corpora[key], self.theta_disease, self.phi_disease[key]) for key in self.corpora_disease.corpora.keys()}
            
            # print([len(z) for z in z_disease for zz in z])
            ## phi
            phi_disease_1 = {key: self.sample_phi(self.corpora_disease.corpora[key], z_disease[key]) for key in self.corpora_disease.corpora.keys()}
            ## theta
            theta_disease_1 = self.sample_theta_drug(z_disease, self.alpha_0_disease)
            
            # record sample
            # disease
            theta_disease_log.append(theta_disease_1)
            for key in phi_disease_log:
                phi_disease_log[key].append(phi_disease_1[key])
            
            # 更新
            self.theta_disease = theta_disease_1
            self.phi_disease = phi_disease_1
        # サンプリング終了 ==========================

        self.theta_disease_log = np.stack(theta_disease_log)
        self.phi_disease_log = {key: np.stack(item) for key, item in phi_disease_log.items()}
        # self.w_disease_log 


    def sample_z(self, corpus: MyCorpus, theta, phi):            
        # estimate p(z) per vocabulary (not word)
        voc_index = []
        for w_d in corpus.corpus:
            voc_index_d = []
            for w_di in w_d:
                voc_index_d.extend([w_di[0]]*w_di[1])
            
            voc_index.append(voc_index_d)
        #voc_index = [Lda.corpus_flatten(w_d) for w_d in corpus.corpus]
        
        p_z = [
            Lda.normalized_inner_join(theta[i, :].reshape((-1, 1)), phi[:, w_d]) 
            for i, w_d in enumerate(voc_index)
        ]
        # sample z from p(z)
        z_sample = [Lda.sample_from_discrete_dist(p_z_d) for p_z_d in p_z]
        
        return z_sample

    def sample_z_parallel(self, corpus: MyCorpus, theta, phi):            
        # estimate p(z) per vocabulary (not word)
        with Pool(processes = 10) as p:
            voc_index = p.map(Lda.corpus_flatten, iterable=corpus.corpus)
        #voc_index = [[w_di[0] for w_di in w_d for i in range(w_di[1])] for w_d in corpus.corpus]
        
        p_z = [
            Lda.normalized_inner_join(theta[i, :].reshape((-1, 1)), phi[:, w_d]) 
            for i, w_d in enumerate(voc_index)
        ]
        # sample z from p(z)
        z_sample = [Lda.sample_from_discrete_dist(p_z_d) for p_z_d in p_z]
        
        return z_sample

    def sample_phi(self, corpus: MyCorpus, z: List[List]):
        # estimate p(phi|beta\1)
        beta_1 = np.ones([self.N_topic, len(corpus.dictionary)])*self.beta_0
        for z_d, w_d in zip(z, corpus.corpus):
            for i, w_di in enumerate(w_d):
                beta_1[z_d[i]][w_di[0]] = beta_1[z_d[i]][w_di[0]] + w_di[1]
        
        phi = np.array([np.random.dirichlet(beta_1[k]) for k in range(self.N_topic)])

        return phi

    def sample_theta_disease(self, z_disease_full, theta_disease_old, theta_drug, link_drug_disease):
        '''
        sample theta(source) from z(source) and theta(destination) 
        '''
        # concatenate Zs from other sources
        z_disease = [np.concatenate(zz) for zz in zip(*z_disease_full.values())]

        alpha_disease_topic = np.ones([len(z_disease), self.N_topic])*self.alpha_0_disease
        for d, z_d in enumerate(z_disease):
            for i in range(len(z_d)):
                alpha_disease_topic[d, z_d[i]] = alpha_disease_topic[d, z_d[i]] + 1

        # alpha_drug_topic = link_drug_disease.dot(theta_disease_old-self.alpha_0_disease)
        alpha_drug_topic = link_drug_disease.dot(theta_disease_old) + \
            np.sum((1-link_drug_disease), axis=1)[:, np.newaxis]*self.alpha_0_disease
        alpha_drug_topic = alpha_drug_topic/self.N_disease

        theta_disease = np.array([
            self.sample_theta_by_sir(
                alpha_di, 
                alpha_drug_topic, 
                theta_drug, 
                theta_di_old, 
                link_to_drugs)
            for alpha_di, theta_di_old, link_to_drugs in zip(alpha_disease_topic, theta_disease_old, link_drug_disease.transpose())
        ])

        return theta_disease

    def sample_theta_by_sir(
        self, 
        alpha_di: np.ndarray, # (N topic, )
        alpha_drug_topic: np.ndarray, # (N drug, N topic)
        theta_drug: np.ndarray, # (N drug, N topic)
        theta_di_old: np.ndarray, # (N topic, ) 
        link_to_drug: np.ndarray, # (N drug, )
        ):

        # the number of samples at 1st sampling
        N = np.min([np.power(10, len(alpha_di)), int(100)])

        theta_sampled = np.random.dirichlet(np.ones_like(theta_di_old), N)
        alpha_drug_sampled0 = link_to_drug[:, np.newaxis, np.newaxis]*(theta_sampled - theta_di_old).transpose()[np.newaxis]/self.N_disease
        alpha_drug_sampled = alpha_drug_topic[:, :, np.newaxis] + alpha_drug_sampled0

        # print(f'alpha_drug_sampled: {alpha_drug_sampled}')

        tmp1_di = loggamma(np.sum(alpha_drug_sampled, axis=1)) + \
            np.sum((alpha_drug_sampled - 1)*np.log(theta_drug[:, :, np.newaxis]) - loggamma(alpha_drug_sampled), axis=1)
            # (N drug, N sample)
        tmp2_disease_dirichlet=np.sum((alpha_di-1)*np.log(theta_sampled), axis=1) # (N sample, )
        logpdf_p_tilde = np.sum(tmp1_di, axis=0) + tmp2_disease_dirichlet
        
        # print(f'logp(theta): {logpdf_p_tilde}')
        log_w = logpdf_p_tilde - logsumexp(logpdf_p_tilde)
        w = np.exp(log_w)
        
        sampled_index = np.random.choice(np.arange(N), p=w)
        
        return theta_sampled[sampled_index, :]

    def sample_theta_drug(self, z_full: Dict[str, List], alpha_0 = None) -> np.ndarray:
        # concatenate Zs from other sources 
        z = [np.concatenate(zz) for zz in zip(*z_full.values())]

        if alpha_0 is None:
            alpha_0 = self.alpha_0_drug
        alpha_1 = np.ones([len(z), self.N_topic])*alpha_0
        for d, z_d in enumerate(z):
            for i in range(len(z_d)):
                alpha_1[d, z_d[i]] = alpha_1[d, z_d[i]] + 1

        theta = np.array([np.random.dirichlet(alpha) for alpha in alpha_1])

        return theta

    def estimate_params(self, n=None):
        if n is None:
            n = 0
        
        res = dict()
        # theta 
        if 'theta_disease_log' in dir(self):
            theta_disease_hat = np.mean(self.theta_disease_log[n:, :, :], axis=0)
            df_theta_disease_hat = pd.DataFrame(
                theta_disease_hat,
                index = self.corpora_disease.title,
                columns = [f'Topic:{i:02}' for i in range(self.N_topic)]
            )
            res.update(disease = {'theta': df_theta_disease_hat})
        if 'theta_drug_log' in dir(self):
            theta_drug_hat = np.mean(self.theta_drug_log[n:, :, :], axis=0)
            df_theta_drug_hat = pd.DataFrame(
                theta_drug_hat,
                index = self.corpora_drug.title,
                columns = [f'Topic:{i:02}' for i in range(self.N_topic)]
            )
            res.update(drug = {'theta': df_theta_drug_hat})
        
        # phi
        if 'phi_disease_log' in dir(self):
            phi_disease_hat = {key: np.mean(self.phi_disease_log[key][n:, :, :], axis=0) for key in self.corpora_disease.corpora.keys()}
            df_phi_disease_hat = {
                key: pd.DataFrame(
                    phi_disease_hat[key].T,
                    index=[self.corpora_disease.corpora[key].dictionary[i] for i in range(len(self.corpora_disease.corpora[key].dictionary))],
                    columns=[f'Topic:{i:02}' for i in range(self.N_topic)]
                ) for key in self.corpora_disease.corpora.keys()
            }
            if 'disease' in res:
                res['disease'].update(phi = df_phi_disease_hat)
            else:
                res.update(disease = {'phi': df_phi_disease_hat})
        if 'phi_drug_log' in dir(self):
            phi_drug_hat = {key: np.mean(self.phi_drug_log[key][n:, :, :], axis=0) for key in self.corpora_disease.corpora.keys()}
            df_phi_drug_hat = {
                key: pd.DataFrame(
                    phi_drug_hwat[key].T,
                    index=[self.corpora_drug.corpora[key].dictionary[i] for i in range(len(self.corpora_drug.corpora[key].dictionary))],
                    columns=[f'Topic:{i:02}' for i in range(self.N_topic)]
                ) for key in self.corpora_drug.corpora.keys()
            }
            if 'drug' in res:
                res['drug'].update(phi = df_phi_drug_hat)
            else:
                res.update(drug = {'phi': df_phi_drug_hat})
        
        return res

    def calc_perplexity(self, used=None):
        w_li = {key: 
            np.hstack(Lda.calc_word_likelihood(
                self.corpora_disease.corpora[key],
                self.theta_disease_log,
                self.phi_disease_log[key],
                used
            ))
            for key in self.corpora_disease.corpora.keys()
        }

        perps_per_data = {key: gmean(1/li) for key, li in w_li.items()}
        perp = gmean(1/np.hstack([d for d in w_li.values()]))
        
        return perp, perps_per_data

    @staticmethod
    def calc_word_likelihood(corpus: MyCorpus, theta_log, phi_log, used=None):
        if used is None:
            used = theta_log.shape[0]
        
        voc_index = []
        for w_d in corpus.corpus:
            voc_index_d = []
            for w_di in w_d:
                voc_index_d.extend([w_di[0]]*w_di[1])
            
            voc_index.append(voc_index_d)
        
        theta_used = theta_log[-used:, :, :]
        phi_used = phi_log[-used:, :, :]
        
        w_li = [
            np.array([
                (np.squeeze(phi_used[j, :, w_d])).dot(np.squeeze(theta_used[j, i, :])) 
                for j in range(phi_used.shape[0])
            ]).mean(axis=0)
            for i, w_d in enumerate(voc_index)]        
        return w_li

    @staticmethod
    def corpus_flatten(w_d):
        voc_index_d = []
        for w_di in w_d:
            voc_index_d.extend([w_di[0]]*w_di[1])

        return voc_index_d

    @staticmethod
    def normalized_inner_join(a, b):
        """ a's shape is (N, 1), b's shape is (N, M)"""
        c = a*b
        return c/np.sum(c, axis=0)

    @staticmethod
    def sample_from_discrete_dist(mat_p):
        n_topic, n_voc = mat_p.shape
        
        u = np.random.rand(n_voc)
        cdf = np.cumsum(mat_p, axis=0)
        z = np.zeros(n_voc, dtype=np.int64)
        for i in range(n_voc):
            for k in range(n_topic):
                if u[i] >= cdf[k, i]:
                    z[i] = z[i]+1
                else:
                    break

        return z


if __name__ == '__main__':
    pass
