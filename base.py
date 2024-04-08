from abc import ABC, abstractmethod

import os
from datetime import datetime

import pandas as pd


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
    
    
        
    
        
    