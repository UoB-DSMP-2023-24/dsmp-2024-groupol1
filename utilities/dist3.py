import os
import time

import pandas as pd
import numpy as np
from pandas import DataFrame

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from tcrdist.repertoire import TCRrep

from utilities.base import BaseClass
from utilities.utils import visualise_data



class Dist3(BaseClass):
    def __init__(self, input_data: DataFrame) -> None:
        super().__init__(input_data)
        self._output_dir = os.path.join(os.getcwd(), 'visualisations')
        
    def _preprocess_data(self):
        _selected_features = self._input_data[['gene','cdr3','v.segm','j.segm','species','mhc.a','mhc.b','mhc.class','antigen.epitope','antigen.species','vdjdb.score', 'complex.id']]
        _human_data = _selected_features[(_selected_features['species'] == 'HomoSapiens') & (_selected_features['vdjdb.score'] > 0)]

        # Drop duplicate rows
        _human_data_cols = _human_data.columns.difference(['complex.id', 'vdjdb.score'])
        _human_data = _human_data.drop_duplicates(subset=_human_data_cols)

        # Delete rows with null values=
        _human_data.dropna(inplace = True)
        _TRA = _human_data[_human_data['gene'] =='TRA']
        _TRA.rename(columns={'cdr3':'cdr3_a_aa',
                                     'v.segm':'v_a_gene', 
                                     'j.segm':'j_a_gene'}, 
                            inplace=True)

        _TRB = _human_data[_human_data['gene'] =='TRB']
        _TRB.rename(columns={'cdr3':'cdr3_b_aa',
                             'v.segm':'v_b_gene', 
                             'j.segm':'j_b_gene'}, 
                    inplace=True)

        if self._chains == ['beta']:
          self._processed_data = _TRB
        elif self._chains == ['alpha']:
          self._processed_data = _TRA
        else:
          _temp = pd.merge(_TRA[['cdr3_a_aa', 'v_a_gene', 'j_a_gene', 'mhc.a', 'antigen.epitope', 'vdjdb.score', 'complex.id']].reset_index(drop = True),
                           _TRB[['cdr3_b_aa', 'v_b_gene', 'j_b_gene', 'mhc.b', 'complex.id']].reset_index(drop = True),
                           how = 'inner',
                           on = 'complex.id')
          
          self._processed_data = _temp[_temp['complex.id'] != 0]

          

        
    def run_model(self, chains = ['beta']):
        self._chains = chains
        self._preprocess_data()
        
        start_time = time.time()
        
        tr = TCRrep(cell_df = self._processed_data,
                    organism = 'human',
                    chains = chains,
                    db_file = 'alphabeta_gammadelta_db.tsv')
        
        if (len(chains)==1) and (chains[0] == 'beta'):
            distance_matrix = pd.concat([pd.DataFrame(tr.pw_cdr3_b_aa), tr.clone_df['antigen.epitope']], axis = 1)
        elif (len(chains)==1) and (chains[0] == 'alpha'):
            distance_matrix = pd.concat([pd.DataFrame(tr.pw_cdr3_a_aa), tr.clone_df['antigen.epitope']], axis = 1)
        else:
            matrix = np.hstack((tr.pw_cdr3_b_aa, tr.pw_cdr3_a_aa))
            distance_matrix = pd.concat([pd.DataFrame(matrix), tr.clone_df['antigen.epitope']], axis = 1)
            
        self._tcr_dist_rep = distance_matrix
 
        end_time = time.time()
        self._settings['time_to_run'] = end_time - start_time           
            
    def reduce_dimensionality(self, title):   
        _value_counts_antigen = self._tcr_dist_rep['antigen.epitope'].value_counts()
        if self._chains == ['alpha', 'beta']:
          list_of_eps = ['ATDALMTGY', 'ELAGIGILTV', 'GILGFVFTL', 'GLCTLVAML',  'NLVPMVATV', 'RAKFKQLL']
        else:
          list_of_eps = ['ATDALMTGY', 'ELAGIGILTV', 'GILGFVFTL', 'GLCTLVAML', 'KRWIILGLNK', 'NLVPMVATV', 'RAKFKQLL']
        
        distance_matrix_filtered = self._tcr_dist_rep[self._tcr_dist_rep['antigen.epitope'].isin(list_of_eps)]
        
        PCA_model = PCA(n_components = 50)
        _embedding_pca = PCA_model.fit_transform(distance_matrix_filtered.iloc[:, :-1].values)
        #apply TSNE on 50 components.
        TSNE_model = TSNE(n_components=2, perplexity=30.0)
        dist_reduced = TSNE_model.fit_transform(_embedding_pca)
        
        visualise_data(dist_reduced, distance_matrix_filtered['antigen.epitope'], self._output_dir, title = title)
        
        self._t_cells_reduced = pd.DataFrame(dist_reduced)
        self._t_cells_reduced['Epitope'] = distance_matrix_filtered['antigen.epitope'].tolist()   

    def record_performance(self):
        return self._cluster_data()
        
        
        

                                                                                                   
                                                            
                
        
