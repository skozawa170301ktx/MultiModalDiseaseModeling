""" A set of functions created to predict TI/SE

Methods :
1. When a target is presented in the form of a KEGG ID, the hsa map associated with it is retrieved.
2. Build that hsa map as a network and search for relevant Paths.
3. Calculate the Levenshtein ratio with the feature Paths, and for each feature, get the vector of (1 - Levenshtein ratio).
4. Input the matrix into each TI/SE model and produce the result. (Threshhold should be modifiable.)

"""


import pandas as pd
import numpy as np
import networkx as nx
import itertools
from scipy.sparse import csr_matrix
import pickle
from scipy.sparse import load_npz

def levenshtein_distance(x):
    """ A program to calculate the Levenshtein ratio from two paths
    Args:
        x (tuple): A tuple containing two paths
        
    Returns:
        float: Calculated value of Levenshtein ratio

    """
    a = x[0]
    b = x[1]
    m = np.zeros((len(a) + 1, len(b) + 1), np.int32)
    for i in range(len(a) + 1):
        m[i,0] = i
    for j in range(len(b) + 1):
        m[0,j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                x = 0
            else:
                x = 1
            m[i,j] = min(m[i - 1,j] + 1, m[i, j - 1] + 1, m[i - 1,j - 1] + x)
    return m[-1,-1] / max(len(a), len(b))


def leven_without_1(series):
    """ A program that returns 'NaN' if there are no duplicate path elements in the two paths
    Args:
        series (tuple): A tuple containing two paths
        
    Returns:
        series (tuple): input
        or
        str: Return 'NaN' if there are no duplicates

    """
    main_path = series[0]
    sub_path = series[1]
    if len(set(main_path)&set(sub_path))==0:
        return 'NaN'
    else:
        return series


def build_network(KEGG_ID, kegg_pathwayid, df_node, df_edge):
    """ A program to build a network from xml files that can be downloaded from KEGG
    Args:
        kegg_pathwayid (str): Targeting hsa map ID
        
    Returns:
        pd.DataFrame: Dataframe about paths extracted from network

    """
    if type(KEGG_ID) == str:
        if KEGG_ID[0] == 'C':
            kind_id = 'cpd:'
        elif KEGG_ID[0] == 'D':
            kind_id = 'dr:'
        elif KEGG_ID[0] == 'G':
            kind_id = 'gl:'
        else:
            kind_id = 'hsa:'
    else:
        kind_id = 'hsa:'

    df_network = pd.DataFrame()
    df_no = df_node[df_node['hsa']==kegg_pathwayid]
    df_ed1 = df_edge[df_edge['hsa']==kegg_pathwayid]
        
    a1 = df_no[df_no['type']=='group']
    # Convert a list of edges from the ID of the elements of a group to the ID of the whole group.
    df_ed = df_ed1.copy()
    for i, k in zip(a1['index'],a1['component']):
        k = list(map(int, k))
        df_ed = df_ed.replace(k, i)
        
    df = pd.merge(df_ed,df_no, left_on='entry1', right_on='index', how = 'left')
    df = pd.merge(df,df_no, left_on='entry2', right_on='index', how = 'left')
    G_g = nx.DiGraph()
    df_g = df.copy()
        
    list_node_g = list(set(list(df_g['entry1']) + list(df_g['entry2'])))
    G_g.add_nodes_from(list_node_g)
        
    list_edge_g = []
    for i in range(len(df_g)):
        list_edge_g.append(tuple(df_g[['entry1','entry2']].loc[i]))
    G_g.add_edges_from(list_edge_g)
        
    end_g = []  # Add the endpoint node to 'end_g'.
    for i in list(G_g.nodes):
        if G_g.out_degree(i) == 0:
            end_g.append(i)
        else:
            pass
        
    a2 = df_no.copy() # In 'df_no', convert the ID of an element of a group to the ID of the whole group.
    for i, k in zip(a1['index'],a1['component']):
        k = list(map(int, k))
        a2 = a2.replace(k, i)
            
    num = list(a2[a2['KEGG_id_1'].str.contains(kind_id + str(KEGG_ID) + ' ')]['index'].values) # Add the starting node to 'num'.
        
    for o in num:
        try:
            for i in end_g:
                if len(list(nx.all_simple_paths(G_g, source=o, target = i)))!=0:
                    s = list(nx.all_simple_paths(G_g, source=o, target = i))
                    df_network = pd.concat([df_network, pd.DataFrame([[KEGG_ID,kegg_pathwayid,s]],columns = ['HSA','hsa_map','order'])])
                else:
                    pass
        except:
            pass
            
    df_network = df_network.reset_index(drop = True)
    return df_network

def exclude_hsa_map(df, map_list):
    df1 = df.copy()
    for i in map_list:
        df1 = df1[df1['hsa'] != i]
    return df1
    
def PATHs_Search(KEGG_ID):
    """ Programs to search for Paths related to Target
    Args:
        KEGG_ID (int): KEGG ID of Target
        
    Returns:
        PATH_list (list): List of Paths associated with Target

    """
    if type(KEGG_ID) == str:
        if KEGG_ID[0] == 'C':
            kind_id = 'cpd:'
        elif KEGG_ID[0] == 'D':
            kind_id = 'dr:'
        elif KEGG_ID[0] == 'G':
            kind_id = 'gl:'
        else:
            kind_id = 'hsa:'
    else:
        kind_id = 'hsa:'

    # 1. When a target is presented in the form of a KEGG ID, the hsa map associated with it is retrieved.
    df_edge = pd.read_csv('../1_Paths_from_KEGG_Pathway_code/output/all_target_edge_DrugBank.csv',header = 0, index_col=0)
    df_node = pd.read_csv('../1_Paths_from_KEGG_Pathway_code/output/all_target_node_DrugBank.csv',header = 0, index_col=0)
    df_n2 = pd.read_csv('../1_Paths_from_KEGG_Pathway_code/output/all_target_node_DrugBank_group.csv',header = 0, index_col=0)
    
    map_list = ['hsa00230', 'hsa00061']
    df_edge = exclude_hsa_map(df_edge, map_list)
    df_node = exclude_hsa_map(df_node, map_list)
    df_n2 = exclude_hsa_map(df_n2, map_list)

    # Convert the elements of the component column whose type is "group" from the string type to the list type and its contents to the int type.
    df_no1 = df_node[df_node['type']=='group']['component'].apply(eval)
    df_n = pd.merge(df_node, pd.DataFrame(df_no1),left_index=True, right_index=True ).drop(columns='component_x').rename(columns = {'component_y':'component'})
    df_node = pd.concat([df_node[df_node['type']!='group'], df_n])

    df_node['KEGG_id_1'] = df_node['KEGG_id'] + ' '

    HSA_list = list(set(df_node[df_node['KEGG_id_1'].str.contains(kind_id + str(KEGG_ID) + ' ')]['hsa']))
    
    # 2. Build that hsa map as a network and search for relevant Paths.

    df_all1 = pd.DataFrame()
    for HSA_list_ in HSA_list:
        df1 = build_network(KEGG_ID, HSA_list_, df_node, df_edge)
        df_all1 = pd.concat([df_all1, df1])
    df_all1 = df_all1.reset_index(drop = True)
    
    df_all_unstack = df_all1['order'].apply(pd.Series).unstack().reset_index().sort_values(['level_1','level_0']).dropna(subset=[0]).reset_index(drop = True)
    df_a = pd.merge(df_all_unstack,df_all1,left_on='level_1',right_index = True).drop(columns = 'order').rename(columns = {0:'order'}).reset_index(drop = True).drop(columns = ['level_0','level_1']).reset_index()

    df_a_unstack = df_a['order'].apply(pd.Series).unstack().reset_index().sort_values(['level_1','level_0']).dropna(subset=[0]).reset_index(drop = True)

    df_b = pd.merge(df_a_unstack,df_a,left_on='level_1',right_on ='index').drop(columns = ['order', 'level_1']).rename(columns = {'level_0':'index_m'}).rename(columns = {0:'order'})
    df_b['order'] = df_b['order'].astype(int)
    df = pd.DataFrame()
    for i in list(set(df_b['hsa_map'])):
        f = pd.merge(df_b[df_b['hsa_map']==i], df_n2[df_n2['hsa']==i], left_on = 'order',right_on = 'ID', how = 'left')
        df = pd.concat([df, f])

    df.index = list(df['index_m'])
    df = df[['index', 'HSA', 'hsa_map', 'KEGG_id']].groupby(['index', 'HSA', 'hsa_map'])['KEGG_id'].apply(lambda x: [s for s in x.values.tolist()]).reset_index()

    # Exclude duplicate Paths.
    df['KEGG_id'] = df['KEGG_id'].astype(str)
    df = df.drop_duplicates(subset = 'KEGG_id')
    df['KEGG_id'] = df['KEGG_id'].apply(eval)
    PATH_list = list(df['KEGG_id'])

    return PATH_list


def PCA_TI(Levenshtein_matrix, TI_ID):
    """ Program to sample the features for each TI and perform PCA (Threshhold can be changed)
    Args:
        Levenshtein_matrix (csr_matrix): A matrix that the value is calclated by (1 - Levenshtein ratio)
        
    Returns:
        PCA_matrix (csr_matrix): A matrix converted by PCA

    """
    # 4. Input the vector of maxima into each SE model and produce the result. (Threshhold should be modifiable.)
    
    y = pd.DataFrame(load_npz('../9_Integration_SE_TI_Target_datafile/Y_binary_TI.npz').toarray())
    Levenshtein_matrix = Levenshtein_matrix[:, list(y[y[TI_ID]==1].index)]
    with open('../6_PCA_model/X_PCA_model_TI_'+ str(TI_ID) +'.pkl', 'rb') as ti:
        pca_model_ti = pickle.load(ti)

    PCA_matrix = csr_matrix(pca_model_ti.transform(Levenshtein_matrix.toarray()))

    return PCA_matrix


def LGBM_TI(PCA_matrix, TI_ID, TI_Name ,Threshhold):
    """ Program that makes predictions using a matrix converted by PCA
    Args:
        PCA_matrix (csr_matrix): A matrix converted by PCA
        Threshhold (float): Threshold for prediction
        
    Returns:
        df_TI (DataFrame): Dataframe of predicted TI

    """

    df_TI = pd.DataFrame()

    try:
        y_preds = []
        for k in range(10):

            m = pd.read_pickle('../11_LGBM_TI_PCA/model_ti_' + str(TI_ID) + '_' + str(k) + '.pkl')
            y_pred = m.predict(PCA_matrix, num_iteration = m.best_iteration)
            pred = list(np.where(y_pred < Threshhold, 0, 1))
            y_preds.append(pred)
        y_pred_new = np.where(np.average(y_preds, axis = 0) <= 0.5, 0, 1)
        scores = pd.DataFrame([y_pred_new.sum()/len(y_pred_new)], columns = [TI_Name])

        df_TI = pd.concat([df_TI, scores], axis = 1).T.rename(columns = {0:'TI'})
    except:
        pass

    return df_TI

def TI_Prediction_From_TARGET(df_KEGG_ID, Threshhold = 0.5, TI_list = None):
    """ A main program that makes TI predictions using Paths starting from Target as input values
    Args:
        df_KEGG_ID (DataFrame): Dataframe of KEGG ID
        Threshhold (float): Threshold for prediction
        
    Returns:
        df_TI (DataFrame): Dataframe of predicted TI

    """
    
    df_use = pd.read_csv('../9_Integration_SE_TI_Target_datafile/Y_ID_name_TI.csv',header = 0, index_col=0).reset_index()
    ti_use = pd.read_csv('../4_Feature_extraction/output/Train_Test_count_TI.csv', header = 0, index_col=0)
    ti_use = ti_use[ti_use['use_ID']==1]
    df_use = pd.merge(ti_use, df_use, left_on = 'ID', right_on = 'index').drop(columns = 'index')
    if TI_list is None:
        pass
    else:
        df_use = pd.merge(pd.DataFrame(TI_list), df_use, left_on = 0, right_on = 'TI_name')

    df_all_Target = df_use[['TI_name']]
    for KEGG_ID in df_KEGG_ID['KEGG_ID']:
        try:
            df_Target = pd.DataFrame()
            PATH_list = PATHs_Search(KEGG_ID)
            Levenshtein_matrix = Levenshtein_calc(PATH_list)
            for TI_ID, TI_Name in zip(df_use['ID'], df_use['TI_name']):
                
                PCA_matrix = PCA_TI(Levenshtein_matrix, TI_ID)
                df_TI = LGBM_TI(PCA_matrix, TI_ID, TI_Name, Threshhold).rename(columns = {'TI':KEGG_ID})
                df_Target = pd.concat([df_Target, df_TI])
            df_all_Target = pd.merge(df_all_Target, df_Target, left_on= 'TI_name', right_index=True)
            
        except:
            print(f'Error in TI : KEGG_ID {KEGG_ID}')
    return df_all_Target.set_index('TI_name')

def Levenshtein_calc(PATH_list):
    """ A program that calculates the Levenshtein ratio with feature Paths and obtains the vector of (1 - Levenshtein ratio) for each feature
    Args:
        PATH_list (list): List of Paths associated with Target
        
    Returns:
        Levenshtein_matrix (csr_matrix): A matrix that the value is calclated by (1 - Levenshtein ratio)

    """
    # 3. Calculate the Levenshtein ratio with the feature Paths, and for each feature, get the vector of maximum values of (1 - Levenshtein ratio).
    df_feature = pd.read_csv('../3_Calc_Edit_Distance/output/KEGG_ID_index.csv',header = 0, index_col=0)
    df_feature['KEGG_id'] = df_feature['KEGG_id'].apply(eval)
    use_list_b = list(df_feature['KEGG_id'])
    df_le = pd.DataFrame()
    for i, numb in zip(PATH_list, range(len(PATH_list))):
        use_list_a = [i]
        v = list(itertools.product(use_list_a, use_list_b))
        df_i = pd.DataFrame(v)
        df_i['li'] = v
        df_i['li'] = df_i['li'].apply(leven_without_1)
        df_i_0 = df_i[df_i['li'] != 'NaN']
        df_i_0['Levenshtein_ratio'] = df_i_0['li'].apply(levenshtein_distance)
        a = df_i_0[[0,1,'Levenshtein_ratio']]
        a['number'] = numb
        df_le = pd.concat([df_le, a])

    df_le['value'] = 1 - df_le['Levenshtein_ratio']

    df_le[0] = df_le[0].astype(str)
    df_le[1] = df_le[1].astype(str)

    df_feature['KEGG_id'] = df_feature['KEGG_id'].astype(str)
    df_le1 = pd.merge(df_feature, df_le, left_on = 'KEGG_id', right_on = 1)
    df_le1['id_1'] = df_le1['number'].rank(method='dense') - 1
    Levenshtein_matrix = csr_matrix((list(df_le1['value']), (list(df_le1['id_1']), list(df_le1['index']))), shape=(int(df_le1['id_1'].max()+1), len(df_feature)))
    return Levenshtein_matrix


def PCA_SE(Levenshtein_matrix, SE_ID):
    """ Program to sample the features for each SE and perform PCA (Threshhold can be changed)
    Args:
        Levenshtein_matrix (csr_matrix): A matrix that the value is calclated by (1 - Levenshtein ratio)
        
    Returns:
        PCA_matrix (csr_matrix): A matrix converted by PCA

    """
    # 4. Input the matrix into each SE model and produce the result. (Threshhold should be modifiable.)
    
    y = pd.DataFrame(load_npz('../9_Integration_SE_TI_Target_datafile/Y_binary_SE.npz').toarray())

    Levenshtein_matrix = Levenshtein_matrix[:, list(y[y[SE_ID]==1].index)]
    with open('../6_PCA_model/X_PCA_model_SE_'+ str(SE_ID) +'.pkl', 'rb') as se:
        pca_model_se = pickle.load(se)

    PCA_matrix = csr_matrix(pca_model_se.transform(Levenshtein_matrix.toarray()))

    return PCA_matrix


def LGBM_SE(PCA_matrix, SE_ID, SE_Name ,Threshhold):
    """ Program that makes predictions using a matrix converted by PCA
    Args:
        PCA_matrix (csr_matrix): An matrix converted by PCA
        Threshhold (float): Threshold for prediction
        
    Returns:
        df_SE (DataFrame): Dataframe of predicted SE

    """

    df_SE = pd.DataFrame()

    try:
        y_preds = []
        for k in range(10):

            m = pd.read_pickle('../11_LGBM_SE_PCA/model_se_' + str(SE_ID) + '_' + str(k) + '.pkl')
            y_pred = m.predict(PCA_matrix, num_iteration = m.best_iteration)
            pred = list(np.where(y_pred < Threshhold, 0, 1))
            y_preds.append(pred)
        y_pred_new = np.where(np.average(y_preds, axis = 0) <= 0.5, 0, 1)
        scores = pd.DataFrame([y_pred_new.sum()/len(y_pred_new)], columns = [SE_Name])

        df_SE = pd.concat([df_SE, scores], axis = 1).T.rename(columns = {0:'SE'})
    except:
        pass

    return df_SE

def SE_Prediction_From_TARGET(df_KEGG_ID, Threshhold = 0.5, SE_list = None):
    """ A main program that makes SE predictions using Paths starting from Target as input values
    Args:
        df_KEGG_ID (DataFrame): Dataframe of KEGG ID
        Threshhold (float): Threshold for prediction
        
    Returns:
        df_TI (DataFrame): Dataframe of predicted SE

    """
    
    df_use = pd.read_csv('../9_Integration_SE_TI_Target_datafile/Y_ID_name_SE.csv',header = 0, index_col=0).reset_index()
    se_use = pd.read_csv('../4_Feature_extraction/output/Train_Test_count_SE.csv', header = 0, index_col=0)
    se_use = se_use[se_use['test']!=0]
    df_use = pd.merge(se_use, df_use, left_on = 'ID', right_on = 'index').drop(columns = 'index')
    if SE_list is None:
        pass
    else:
        df_use = pd.merge(pd.DataFrame(SE_list), df_use, left_on = 0, right_on = 'SE_name')

    df_all_Target = df_use[['SE_name']]
    for KEGG_ID in df_KEGG_ID['KEGG_ID']:
        try:
            df_Target = pd.DataFrame()
            PATH_list = PATHs_Search(KEGG_ID)
            Levenshtein_matrix = Levenshtein_calc(PATH_list)

            for SE_ID, SE_Name in zip(df_use['ID'], df_use['SE_name']):
                PCA_matrix = PCA_SE(Levenshtein_matrix, SE_ID)
                df_SE = LGBM_SE(PCA_matrix, SE_ID, SE_Name, Threshhold).rename(columns = {'SE':KEGG_ID})
                df_Target = pd.concat([df_Target, df_SE])
            df_all_Target = pd.merge(df_all_Target, df_Target, left_on = 'SE_name', right_index=True)
            
        except:
            print(f'Error in SE : KEGG_ID {KEGG_ID}')
    return df_all_Target.set_index('SE_name')


def SE_TI_Prediction_From_TARGET(KEGG_ID_i, Threshhold = 0.5, max_PATH = 2500, SE_list = None, TI_list = None):
    """ A main program that makes SE/TI predictions using Paths starting from Target as input values
    Args:
        df_KEGG_ID (int): KEGG ID
        Threshhold (float): Threshold for prediction
        max_PATH (int): Upper limit on the number of paths to be extracted
        SE_list (list): Types of SEs to predict
        TI_list (list): Types of TIs to predict
        
    Returns:
        s, t (DataFrame): Dataframe of predicted SE/TI

    """

    try:
        PATH_list = PATHs_Search(KEGG_ID_i)
        if len(PATH_list) > max_PATH:
            print(f'Memory Error in : KEGG_ID {KEGG_ID_i}')
            return '', ''
        else:
            Levenshtein_matrix = Levenshtein_calc(PATH_list)

            df_use = pd.read_csv('../9_Integration_SE_TI_Target_datafile/Y_ID_name_SE.csv',header = 0, index_col=0).reset_index()
            se_use = pd.read_csv('../4_Feature_extraction/output/Train_Test_count_SE.csv', header = 0, index_col=0)
            se_use = se_use[se_use['test']!=0]
            df_use = pd.merge(se_use, df_use, left_on = 'ID', right_on = 'index').drop(columns = 'index')
            if SE_list is None:
                pass
            else:
                df_use = pd.merge(pd.DataFrame(SE_list), df_use, left_on = 0, right_on = 'SE_name')

            df_all_Target = df_use[['SE_name']]
            df_Target = pd.DataFrame()

            for SE_ID, SE_Name in zip(df_use['ID'], df_use['SE_name']):
                PCA_matrix = PCA_SE(Levenshtein_matrix, SE_ID)
                df_SE = LGBM_SE(PCA_matrix, SE_ID, SE_Name, Threshhold).rename(columns = {'SE':KEGG_ID_i})
                df_Target = pd.concat([df_Target, df_SE])
            df_all_Target = pd.merge(df_all_Target, df_Target, left_on = 'SE_name', right_index=True)
            
            
            s = df_all_Target.set_index('SE_name')

            df_use = pd.read_csv('../9_Integration_SE_TI_Target_datafile/Y_ID_name_TI.csv',header = 0, index_col=0).reset_index()
            ti_use = pd.read_csv('../4_Feature_extraction/output/Train_Test_count_TI.csv', header = 0, index_col=0)
            ti_use = ti_use[ti_use['use_ID']==1]
            df_use = pd.merge(ti_use, df_use, left_on = 'ID', right_on = 'index').drop(columns = 'index')
            if TI_list is None:
                pass
            else:
                df_use = pd.merge(pd.DataFrame(TI_list), df_use, left_on = 0, right_on = 'TI_name')

            df_all_Target = df_use[['TI_name']]
            
            df_Target = pd.DataFrame()
            for TI_ID, TI_Name in zip(df_use['ID'], df_use['TI_name']):
                
                PCA_matrix = PCA_TI(Levenshtein_matrix, TI_ID)
                df_TI = LGBM_TI(PCA_matrix, TI_ID, TI_Name, Threshhold).rename(columns = {'TI':KEGG_ID_i})
                df_Target = pd.concat([df_Target, df_TI])
            df_all_Target = pd.merge(df_all_Target, df_Target, left_on= 'TI_name', right_index=True)
            
            t = df_all_Target.set_index('TI_name')
            
            return s, t
    except:
        print(f'Error in : KEGG_ID {KEGG_ID_i}')
        return '', ''