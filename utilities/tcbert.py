import os
import time
from itertools import zip_longest
from typing import Tuple

import torch
import numpy as np
import pandas as pd

import umap

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import BertModel

from utilities.base import BaseClass
from utilities.utils import get_pretrained_bert_tokenizer,  \
                            insert_whitespace, \
                            is_whitespaced, \
                            visualise_data


MODEL_NAME = 'wukevin/tcr-bert'

class BertTcr(BaseClass):
    def __init__(self, 
                 input_data: pd.DataFrame
                 ) -> None:
        super().__init__(input_data)
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._output_dir = os.path.join(os.getcwd(), 'visualisations')
    
    def _load_model_config(self, 
                           name: str
                           ) -> Tuple:
        _model = BertModel.from_pretrained(name, add_pooling_layer=False).to(self._device)
        _tokenizer = get_pretrained_bert_tokenizer(name)
        return _model, _tokenizer
        
    
    def _preprocess_data(self,  
                         species: str = '',
                         antigen_species: str = 'HomoSapiens', 
                         chain_selection: str = 'TRB', 
                         min_vdj_score: int = 1
                        ) -> None:
        
        data = self._input_data
        chain = chain_selection.lower()
        self._settings['Species'] = species
        self._settings['Antigen_species'] = antigen_species
        self._settings['Chain'] = chain
        self._settings['Minimum_VDJ_score'] = min_vdj_score
        data = data[(data['species'] == antigen_species) & (data['vdjdb.score'] > 0) & (data['gene'] == chain_selection)]
        
        data = data[['gene','cdr3','v.segm','j.segm','species','mhc.a','mhc.b','mhc.class','antigen.epitope','antigen.species','vdjdb.score', 'complex.id']]
        
        data_cols = data.columns.difference(['complex.id', 'vdjdb.score'])
        data = data.drop_duplicates(subset=data_cols)
        data.reset_index(inplace = True)
        
        self._processed_data =  data['cdr3'].tolist()
        self._antigen_epitope = data['antigen.epitope']
    
        
    def run_model(self, 
                  n_layer: int = -1
                  ) -> None:
        
        self._preprocess_data()
        model, model_tokenizer = self._load_model_config(MODEL_NAME)
        chunks = [s if is_whitespaced(s) else insert_whitespace(s) for s in self._processed_data]
        
        chunks_pair = [None]
        chunks_zipped = list(zip_longest(chunks, chunks_pair))
        embeddings = []
        
        start_time = time.time()
        #compute forward pass of model
        with torch.no_grad():
            for i, seq_chunk in enumerate(chunks_zipped):
                #tokenize input sequence
                encoded = model_tokenizer(
                    *seq_chunk, padding="max_length", max_length=64, return_tensors="pt"
                )
                
                #move input to GPU
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                #forward pass
                x = model.forward(**encoded, output_hidden_states=True, output_attentions=True)

                # Select the l-th hidden layer for the i-th example
                h = (x.hidden_states[n_layer][0].cpu().numpy().astype(np.float64))
                embeddings.append(h.mean(axis=0))
               
    
        if len(embeddings[0].shape) == 1:
            embeddings = np.stack(embeddings)
        else:
            embeddings = np.vstack(embeddings)
            
        self._t_cell_rep = embeddings
        end_time = time.time()
        self._settings['time_to_run'] = end_time - start_time
        
    
    def reduce_dimensionality(self, title) -> None:
        
        if (self._t_cell_rep is None) or (self._processed_data is None):
            raise ValueError('Beta chain and embeddings not found: please first run the model')
            
        _embedding_df = pd.concat([pd.DataFrame(self._t_cell_rep), 
                                   self._antigen_epitope], 
                                  axis = 1)
        
        #we will only visualise the top 7 most commonly occuring antigens in dataset else visualisation would be uninterpretable
        list_eps = ['ATDALMTGY', 'ELAGIGILTV', 'GILGFVFTL', 'GLCTLVAML', 'KRWIILGLNK', 'NLVPMVATV', 'RAKFKQLL']
  
        _embedding_df_filtered = _embedding_df[_embedding_df['antigen.epitope'].isin(list_eps)]


        PCA_model = PCA(n_components = 50)
        _embedding_pca = PCA_model.fit_transform(_embedding_df_filtered.iloc[:, :-1].values)
        #apply TSNE on 50 components.
        TSNE_model = TSNE(n_components=2, perplexity=30.0)
        dist_reduced = TSNE_model.fit_transform(_embedding_pca)
        
        visualise_data(dist_reduced, _embedding_df_filtered['antigen.epitope'], self._output_dir, title)
        
        self._t_cells_reduced = pd.DataFrame(dist_reduced, 
                                             columns = ['Component 1', 'Component 2'])
        self._t_cells_reduced['Epitope'] = _embedding_df_filtered['antigen.epitope'].tolist()
             
             
    def record_performance(self):
        return self._cluster_data() 
        
        
        
        
                
        
        
        