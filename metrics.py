import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, multilabel_confusion_matrix


def get_purity(df: pd.DataFrame) -> dict:
    '''Compute cluster purity
    :param df: input data
    :type df: pandas DataFrame
    :return: total metrics, epitope-secific metrics and cluster statistics
    :rtype: dict'''

    df.loc[:,'epitope']=df['epitope'].replace([np.nan,'',' '],'None')   # Recode NaNs
    baseline = {ep:len(df[df['epitope']==ep])/len(df) for ep in df['epitope'].unique()} # Get baseline frequence for eadch epitope
    
    mapper ={}
    purity = {}
    purity_enriched = {}
    frequency = {}
    enriched ={}
    N= {}
    Nd = 0
    clusters = df['cluster'].value_counts().index   # Rank clusters
    for c in  clusters:
        t = df[df['cluster']==c]    
        N[c]=len(t)
        if len(t)>0:
            ep=t['epitope'].value_counts().index[0] # Most frequent epitope
            freq = {ep:(len(t[t['epitope']==ep])/len(t)) for ep in t['epitope'].unique()}   # Get frequency of each epitope 
            enrich = {k:(v/baseline[k]) for k,v in freq.items()}    # Compute enrichment of each epitope vs. baseline
            ep_enrich = max(enrich, key=enrich.get) # Find the most enriched
            mapper[c]=ep    # Record the most frequent epitope
            purity[c]=len(t[t['epitope']==ep])/len(t)   # Compute purity for the most frequent epitope
            frequency[c]=freq   # Record the most frequent epitope
            enriched[c]=ep_enrich # Record the most enriched epitope 
            purity_enriched[c]=len(t[t['epitope']==ep_enrich])/len(t)   # Compute purity for the most enriched epitope
            if len(t)<=10:
                Nd+=1   # Find the number of clusters with 10 or fewer members
    
    return {'most_frequent':mapper,
            'most_enriched':enriched,
            'purity_frequent': purity,
            'purity_enriched':purity_enriched,
            'N': N,
            'Nd': Nd}

def precision_recall_fscore(df: pd.DataFrame, ytrue, ypred) -> tuple[float, int, int, float, int, dict[str, int]]:

    labels = np.unique(ytrue)
    precisions = []# per epitope
    recalls = []# per epitope
    accuracies = [] # per epitope
    weights = []# per epitope
    supports = []

    fn_per_epi = df[df['cluster'].isnull()]['epitope'].value_counts() # Retain value counts for false negatives
    
    epmetrics = {'accuracy':{},
                 'precision': {},
                 'recall': {},
                 'f1-score': {},
                 'support': {}}
    for (i, cm) in enumerate(multilabel_confusion_matrix(ytrue, 
                                                         ypred, 
                                                         labels=labels)):# per epitope
        # for order see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
        tn = cm[0][0]
        fn = cm[1][0]
        tp = cm[1][1]
        fp = cm[0][1]
        lbl = labels[i]

        missing_fn = fn_per_epi.get(lbl, 0) 
        fn += missing_fn

        if tp+fp == 0:
            precision = 0.0
        else:
            precision = tp/(tp+fp)

        if tp+fn == 0:
            recall = 0.0
        else:
            recall = tp/(tp+fn)
        
        if tp + tn == 0:
            accuracy = 0
        else:
            accuracy = (tp + tn ) / (tp + tn + fp + fn)


        # weighting='average'
        support = sum(ytrue == lbl)
        w = support/ytrue.shape[0]
        weights.append(w)
        
        precision *= w
        recall *= w
        accuracy *=w

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        supports.append(support)

        epmetrics['accuracy'][labels[i]]= accuracy
        epmetrics['precision'][labels[i]]= precision
        epmetrics['recall'][labels[i]]= recall
        if ((precision * recall ==0)|(precision + recall ==0)):
            f = 0
        else:
            f = 2* (precision * recall) / (precision + recall)

        epmetrics['f1-score'][labels[i]]= f
        epmetrics['support'][labels[i]]=support
                               
    # deal with epitopes for which we had no prediction 
    uncalled_epis = set(df['epitope']).difference(labels)
    if len(uncalled_epis)!=0:
        print('No predictions made for: ',uncalled_epis)
    for i in uncalled_epis:

        accuracy = 0
        precision = 0
        recall = 0
        fscore = 0
        support = sum(ytrue == i)
        recalls.append(recall)
        precisions.append(precision)
        supports.append(sum(ytrue == i))

        epmetrics['accuracy'][i]= accuracy
        epmetrics['precision'][i]= precision
        epmetrics['recall'][i]= recall
        epmetrics['f1-score'][i]= fscore
        epmetrics['support'][i]=support

    recall = sum(recalls)
    precision = sum(precisions)
    support = sum(supports)

    if ((precision * recall == 0)|(precision + recall == 0)):
        f = 0
    else:
        f = 2* (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f, support, epmetrics

def get_clustermetrics(df: pd.DataFrame):

    '''Compute cluster metrics
    :param df: input data
    :type df: pandas DataFrame
    :return: scores
    :rtype: dict'''

    # Find clustered TCRs
    sub = df[~df['cluster'].isnull()]

    # Compute overall purity metrics
    stats = get_purity(sub)

    # Compute predictive metrics
    ypred = sub['cluster'].map(stats['most_frequent'])
    ytrue = sub['epitope']
    accuracy, precision, recall, f1score, support, epmetrics = precision_recall_fscore(
            df, ytrue, ypred)
    ami = adjusted_mutual_info_score(ytrue, ypred)

    # Compute epitope-specific metrics
    counts = {k:v for k,v in sub['epitope'].value_counts().reset_index().values.tolist()}
    maincluster = {ep: sub[sub['epitope']==ep]['cluster'].value_counts().index[0] for ep in sub['epitope'].unique()} # Find the largest cluster per epitope
    mosfreq=stats['most_frequent']    # Find the most frequent epitope per cluster
    clusts = {ep: '-1' if ep not in list(mosfreq.values()) else [c for c in mosfreq.keys() if mosfreq[c]==ep] for ep in counts.keys()} # Map epitopes to the clusters in which they are most frequent
    puritymap = {ep: 0 if clusts[ep]=='-1' else np.mean([len(sub[(sub['cluster']==c)&(sub['epitope']==ep)])/len(sub[sub['cluster']==c]) for c in clusts[ep]]) for ep in clusts.keys()} # Get purity per epitope
    retmap = {ep: len(df[(df['epitope']==ep)&(~df['cluster'].isnull())])/len(df[df['epitope']==ep]) for ep in counts.keys()} # Get retention scores per epitope
    consistencymap = {ep: len(sub[(sub['epitope']==ep)&(sub['cluster']==maincluster[ep])])/counts[ep] for ep in counts.keys()}  # Get consistency scores per epitope
    epmetrics['consistency']=consistencymap
    epmetrics['retention']=retmap
    epmetrics['purity']=puritymap
    
    return {'purity': np.mean(list(stats['purity_frequent'].values())),           # Purity of all clusters weighted equally (frequency)
            'purity_enriched': np.mean(list(stats['purity_enriched'].values())),  # Purity of clusters (enrichment)
            'retention': len(df[~df['cluster'].isnull()])/len(df), # Proportion of clustered TCRs
            'consistency': np.mean([(consistencymap[ep]*counts[ep])/len(sub) for ep in consistencymap.keys()]), # Proportion of an epitope assigned to a given cluster
            'ami':ami,  # Adjusted mutual information
            'accuracy':accuracy,    # Balanced accuracy
            'precision':precision,  # Precision over all epitopes
            'recall':recall,    # Recall over all epitopes
            'f1-score':f1score, # F1 over all epitopes
            'support':support,  # Support
            'epscores': epmetrics,
            'mean_clustsize': np.mean(list(stats['N'].values())), # Average cluster size
            'small_clusters': stats['Nd'],    # Number of clusters with ≤ 10 members
            }, stats  

def score(df: pd.DataFrame, header: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Get scores for a given set of parameters
    :param df: cluster outputs
    :type df: Pandas DataFrame
    :param header: parameters
    :type header: dict
    :return: all scores, epitope-specific scores and cluster statistics
    :rtype: Pandas DataFrame'''

    clusterscores, clusterstats = get_clustermetrics(df)  # Compute scores
    
    # Prepare dataframes
    c_scores = pd.DataFrame.from_dict(clusterscores,orient='index').T
    head = pd.DataFrame.from_dict(header,orient='index').T
    stats  = pd.DataFrame.from_dict(clusterstats).reset_index().rename(columns={'index':'cluster'})
    stats['Model'] = [header['Model']] * len(stats)
    c_scores=pd.concat([head,c_scores],axis=1).reset_index()

    return c_scores, stats
