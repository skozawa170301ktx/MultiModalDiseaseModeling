from typing import List
import numpy as np
from numpy.core.numeric import zeros_like
from numpy.core.numerictypes import maximum_sctype
import pandas as pd

class Lda:
    alpha_0 = 0.1
    beta_0 = 0.1
    n_sample = 100

    def __init__(self, N_topic, dictionary, corpus, n_sample=100):
        self.N_topic = N_topic
        self.dictionary = dictionary
        self.corpus = corpus
        self.n_sample = n_sample

        # initialize variables
        self.theta_0 = np.ones((len(self.corpus), self.N_topic))/self.N_topic # document x topic
        self.phi_0 = np.ones((self.N_topic, len(self.dictionary)))/len(self.dictionary) # topic x dictionary


    def fit(self, theta_0, phi_0):
        theta_log: List[np.ndarray] = []
        phi_log: List[np.ndarray] = []
        self.z_log: List[List] = []
        for n in range(self.n_sample):
            if divmod(n, 100)[1] == 0:
                print(n)

            z = self.sample_z(self.corpus, theta_0, phi_0)
            theta_1 = self.sample_theta(self.corpus, z)
            phi_1 = self.sample_phi(self.corpus, z)

            # record sample
            self.z_log.append([z])
            theta_log.append(theta_1)
            phi_log.append(phi_1)

            theta_0 = theta_1
            phi_0 = phi_1
        
        self.theta_log = np.stack(theta_log)
        self.phi_log = np.stack(phi_log)

    def sample_z(self, corpus, theta, phi):
        topic_list = list(range(self.N_topic))
        # estimate p(z) per vocabulary (not word)
        p_z = [
            Lda.normalized_inner_join(theta[i, :].reshape((-1, 1)), phi[:, [w_ds[0] for w_ds in w_d]]) 
            for i, w_d in enumerate(corpus)
        ]
        # sample z from p(z)
        z_sample = [Lda.sample_from_discrete_dist(p_z_d) for p_z_d in p_z]

        return z_sample
    
    def sample_phi(self, corpus, z):
        # estimate p(phi|beta\1)
        beta_1 = np.ones([self.N_topic, len(self.dictionary)])*self.beta_0
        for z_d, w_d in zip(z, corpus):
            for i, w_di in enumerate(w_d):
                beta_1[z_d[i]][w_di[0]] = beta_1[z_d[i]][w_di[0]] + w_di[1]
        
        phi = np.array([np.random.dirichlet(beta_1[k]) for k in range(self.N_topic)])

        return phi

    def sample_theta(self, corpus, z):
        alpha_1 = np.ones([len(corpus), self.N_topic])*self.alpha_0
        for d, z_d in enumerate(z):
            for i in range(len(z_d)):
                alpha_1[d, z_d[i]] = alpha_1[d, z_d[i]] + 1

        theta = np.array([np.random.dirichlet(alpha) for alpha in alpha_1])

        return theta

    def estimate_params(self, n=None):
        if (n is None):
            theta_hat = np.mean(self.theta_log, axis=0)
            phi_hat = np.mean(self.phi_log, axis=0)
        else:
            theta_hat = np.mean(self.theta_log[n:, :, :], axis=0)
            phi_hat = np.mean(self.phi_log[n:, :, :], axis=0)

        df_phi_hat = pd.DataFrame(
            phi_hat.T,
            index=[self.dictionary[i] for i in range(len(self.dictionary))],
            columns=[f'Topic:{i:02}' for i in range(self.N_topic)]
        )

        return (theta_hat, df_phi_hat)


    @staticmethod
    def df_chi(corpus, dictionary):
        df = [[d, w_di[0], w_di[1], dictionary[w_di[0]]] for d, w_d in enumerate(corpus) for w_di in w_d]
        return df

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
    import pandas as pd
    from glob import glob
    import MeCab
    from gensim.corpora.dictionary import Dictionary
    from gensim.models import LdaModel

    mt = MeCab.Tagger('-d /usr/lib64/mecab/dic/mecab-ipadic-neologd')
    mt.parse('')

    # 辞書・コーパス作成
    flist = glob('data/text/dokujo-tsushin/*')

    train_texts = []
    for ff in flist:
        with open(ff, 'r') as f:
            text = []
            for line in f:
                node = mt.parseToNode(line.strip())
                while node:
                    fields = node.feature.split(",")
                    if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
                        text.append(node.surface)
                    node = node.next
        train_texts.append(text) 

    dictionary = Dictionary(train_texts)
    corpus = [dictionary.doc2bow(text) for text in train_texts]

    my_lda = Lda(N_topic=3, dictionary=dictionary, corpus=corpus, n_sample=5)
    pz = my_lda.fit(my_lda.theta_0, my_lda.phi_0)

