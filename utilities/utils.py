import os
from typing import List

import pandas as pd
import numpy as np
from math import floor

from transformers import BertModel, BertTokenizer

import matplotlib
import matplotlib.pyplot as plt
from tcrdist.repertoire import TCRrep


#special tokens
PAD = "$"
MASK = "."
UNK = "?"
SEP = "|"
CLS = "*"

def is_whitespaced(seq: str
                   ) -> bool:
    """
    This function detects whether there is whitespace between characters in an input string
    """
    tok = list(seq)
    spaces = [t for t in tok if t.isspace()]
    if len(spaces) == floor(len(seq) / 2):
        return True
    return False

def get_pretrained_bert_tokenizer(path: str
                                  ) -> BertTokenizer:
    """Get the pretrained BERT tokenizer. This is a character level tokenizer of amino acids within the beta chain"""
    tok = BertTokenizer.from_pretrained(
        path,
        do_basic_tokenize=False,
        do_lower_case=False,
        tokenize_chinese_chars=False,
        unk_token=UNK,
        sep_token=SEP,
        pad_token=PAD,
        cls_token=CLS,
        mask_token=MASK,
        padding_side="right",
    )
    return tok



def insert_whitespace(seq: str) -> str:
    """ 
    Inserts whitespace between each amino acid in a beta chain sequence. 
    """
    return " ".join(list(seq))


def calculate_dist_and_umap(df: pd.DataFrame,
                            chains: List[str],
                            gene: str) -> pd.DataFrame:
  """
  Function which calculates the T-Cell pairwise distance metric by alpha or beta
  chain using TCR dist and then uses UMAP to reduce dimensionality and outputs a
  plot coloured by epitope.

  """
  df_filtered = df.drop_duplicates(subset = ['cdr3_b_aa', 'antigen.epitope'])

  tr = TCRrep(cell_df = df_filtered,
            organism = 'human',
            chains = chains,
            db_file = 'alphabeta_gammadelta_db.tsv')

  if (len(chains)==1) and (chains[0] == 'beta'):
    distance_matrix = pd.concat([pd.DataFrame(tr.pw_cdr3_b_aa), tr.clone_df[gene]], axis = 1)
    chains = 'beta'
  elif (len(chains)==1) and (chains[0] == 'alpha'):
    distance_matrix = pd.concat([pd.DataFrame(tr.pw_cdr3_a_aa), tr.clone_df[gene]], axis = 1)
    chains = 'alpha'
  else:
    matrix = np.hstack((tr.pw_cdr3_b_aa, tr.pw_cdr3_a_aa))
    distance_matrix = pd.concat([pd.DataFrame(matrix), tr.clone_df[gene]], axis = 1)
    chains = 'alpha_beta'

  value_counts_antigen = distance_matrix[gene].value_counts()
  top_10_value_counts = value_counts_antigen.nlargest(7)
  distance_matrix_filtered = distance_matrix[distance_matrix[gene].isin(top_10_value_counts.index)]

  distances_reduced = umap.UMAP(n_components = 2, n_neighbors = 100, metric='hellinger').fit(distance_matrix_filtered.iloc[:, :-1].values)

  output_dir = 'visualisations'

  f = umap.plot.points(distances_reduced, labels=distance_matrix_filtered[gene])
  f.set_xlabel('UMAP Dimension 1', fontsize=10)
  f.set_ylabel('UMAP Dimension 2', fontsize=10)
  f.set_title(f'UMAP Visualization of {chains}', fontsize=12)

  # Save the figure
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  f.get_figure().savefig(f'{output_dir}/{chains}_chain_umap.png')

  embedding = pd.DataFrame(distances_reduced.embedding_, columns = ['UMAP 1', 'UMAP 2'])
  embedding['Epitope'] = distance_matrix_filtered['antigen.epitope'].tolist()

  distance_df = tr.clone_df[tr.clone_df[gene].isin(top_10_value_counts.index)]


  return embedding, distance_df



def visualise_data(data, labels, output_dir, title):
  plt.figure(figsize=(12, 6))
  epitope_category = pd.Categorical(labels)
  epitope_codes = epitope_category.codes
  epitope_labels = epitope_category.categories
  num_unique_labels = len(epitope_labels)
  colormap = plt.get_cmap('Set1', num_unique_labels)  
  
  sc = plt.scatter(data[:, 0],
                   data[:, 1],
                   c=epitope_codes,
                   s=3,
                   cmap = colormap)
  
  cbar = plt.colorbar(sc, ticks=np.arange(len(epitope_labels)))
  cbar.set_ticklabels(epitope_labels)


  plt.xlabel('TSNE Dim 1')
  plt.ylabel('TSNE Dim 2')
  plt.title(title)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  plt.savefig(f'{output_dir}/beta_chain_tsne_pca.png')
  plt.show()
    