from abc import ABC, abstractmethod

import os
from datetime import datetime

import pandas as pd

from sklearn.cluster import *
from sklearn.metrics import adjusted_mutual_info_score


class BaseClass(ABC):
    """
    Base class from which Giana, TCR-BERT, and TCRDist inherit
    """
    def __init__(self, 
                 input_data: pd.DataFrame) -> None:
        
        self._input_data = input_data
        self._processed_data = None
        self._save_path = os.path.join(os.getcwd(), 'results')
        self._settings = {
            'Datetime': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'Model': 'TCRDist3',
            }
        
    @abstractmethod
    def _preprocess_data(self):
        """
        Method from which data preprocessing can be recorded
        """
        pass
        
    @abstractmethod
    def record_performance(self):
        """
        Evaluate performance of algorithm used for clustering
        """
        pass
    
    @abstractmethod
    def run_model(self):
        pass
    
    
    def _cluster_data(self):
        _actuals = self._t_cells_reduced['Epitope'].astype('category').cat.codes.tolist()
        km = KMeans(n_clusters=7, random_state=42)
        km.fit_predict(self._t_cells_reduced.iloc[:, :-2])
        _score = adjusted_mutual_info_score(_actuals, 
                                            km.labels_)
        self._settings['adjusted_mutual_information_score'] = _score
        print('Adjusted mutual information score: ', _score)
    
    
        
    
        
    