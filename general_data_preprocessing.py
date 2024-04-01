import pandas as pd


def preprocess_data(
    data: pd.DataFrame, 
    species: str, 
    antigen_species: str,
    min_vdj_score: int,
    ) -> pd.DataFrame:
  # Select the columns we need
  selected_features = data[['gene','cdr3','v.segm','j.segm','species','mhc.a','mhc.b','mhc.class','antigen.epitope','antigen.species','vdjdb.score']]

  # Select all species data
  species_data = selected_features[(selected_features['species'] == species) & (selected_features['vdjdb.score'] >= min_vdj_score)]

  # Filter by antigen species
  if antigen_species is not None:
    antigen_species = antigen_species if isinstance(antigen_species, list) else [antigen_species]
    species_data = species_data[species_data['antigen.species'].isin(antigen_species)]

  # Drop duplicate rows
  species_data = species_data.drop_duplicates()

  # Delete rows with null values
  species_data  = species_data.dropna()

  return species_data

def get_bio(data: pd.DataFrame, chain_selection: str, include_j_region: bool) -> pd.DataFrame:
      if include_j_region:
        print('Getting Bio-ID on V, CDR3, J')
        cols = [f'{x}.{chain_selection}' for x in ['v','cdr3','j']]
      else:
          print('Getting Bio-ID on V, CDR3')
          cols = [f'{x}.{chain_selection}' for x in ['v','cdr3']]
      data.loc[:,'bio'] = ['-'.join(x) for x in data[cols].values.tolist()]
      return data
