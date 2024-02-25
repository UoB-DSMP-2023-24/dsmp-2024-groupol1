import pandas as pd

def concatenate_rows(series):
    return ';'.join(series.astype(str))

def get_alpha_beta_chains(df, column_name):
    df[f'{column_name}_a'] = df[column_name].apply(lambda x: x.split(';')[0])
    df[f'{column_name}_b'] = df[column_name].apply(lambda x: x.split(';')[1])
    df = df.drop(columns = column_name)
    print('hi')
    return df

def preprocess_data(file_path: str):
    data = pd.read_csv(file_path, sep = '\t')
    data.drop(columns = ['antigen.gene', 'reference.id', 'method', 'meta', 'cdr3fix', 'web.method', 'web.method.seq', 'web.cdr3fix.nc', 'web.cdr3fix.unmp'], inplace = True)
    non_transformed_cols = ['complex.id', 'species', 'mhc.a', 'mhc.b', 'mhc.class', 'antigen.epitope', 'antigen.species', 'vdjdb.score']
    data = data.sort_values(by = 'gene')
    agg_dict = {col: 'first' for col in non_transformed_cols}
    agg_dict.update({col: concatenate_rows for col in data.columns if col not in non_transformed_cols})

    data_grouped = data.groupby('complex.id', as_index= False).agg(agg_dict)

    data_grouped = get_alpha_beta_chains(data_grouped, 'cdr3')
    data_grouped = get_alpha_beta_chains(data_grouped, 'v.segm')
    data_grouped = get_alpha_beta_chains(data_grouped, 'j.segm')

    data_grouped = data_grouped.drop(columns = 'gene')
    return data_grouped

if __name__ == '__main__':
    df = preprocess_data('data/vdjdb.txt')

