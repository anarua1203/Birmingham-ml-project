
# Funcion para calcular el % de datos faltantes para cada columnas
def missing_values_table(df):
    import pandas as pd
    # Total valores faltantes
    mis_val = df.isnull().sum()

    # % valores faltantes
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Tabla para presentar datos
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Renombrar columnas
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing ', 1 : '% of Total Values'})

    # Ordena la tabla de mayor faltantes a menor
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print resumen
    print ("El dataframe tiene " + str(df.shape[1]) + " columnas.\n"
    "De las cuales a " + str(mis_val_table_ren_columns.shape[0]) +
    " le faltan datos.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

# Funcion para identificar mejores variables basadas en pruebas estadísticas univariadas
def Feature_Selection_k_highest_scores(df,target,stat):
    import pandas as pd
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif,chi2
    select = SelectKBest(stat)
    select.fit(df.drop([target],axis=1), df[target])
    scores = select.scores_
    feature_scores = pd.DataFrame({'columnas': df.drop([target],axis=1).columns.tolist(),'scores': scores.tolist(),})
    return feature_scores

def plt_pca(df):
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import matplotlib.pyplot as plt
    pca = PCA(df.shape[1]-1)
    projected = pca.fit_transform(MinMaxScaler().fit_transform(df))
    pca_inversed_data = pca.inverse_transform(np.eye(df.shape[1]-1))
    plt.style.use('seaborn')

    plt.figure(figsize = (15, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), '--o')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(np.arange(0, df.shape[1]-1, 1.0))
    plt.tight_layout()
    plt.xticks(rotation=90)
    upper_9= list(filter(lambda x: x > 0.9, list(np.cumsum(pca.explained_variance_ratio_))))
    print(f'90 % variance explained with : {list(np.cumsum(pca.explained_variance_ratio_)).index(upper_9[0])} components')
    return pca.components_

def explained_variance(s, n_top_components):
    '''Calculates the approx. data variance that n_top_components captures.
       :param s: A dataframe of singular values for top components;
           the top value is in the last row.
       :param n_top_components: An integer, the number of top components to use.
       :return: The expected data variance covered by the n_top_components.'''

    start_idx = N_COMPONENTS - n_top_components  ## 33-3 = 30, for example
    # calculate approx variance
    exp_variance = np.square(s.iloc[start_idx:,:]).sum()/np.square(s).sum()

    return exp_variance[0]

def display_component(v, features_list,N_COMPONENTS ,component_num, n_weights=10):
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    # get index of component (last row - component_num)
    row_idx = N_COMPONENTS-component_num

    # get the list of weights from a row in v, dataframe
    v_1_row = pd.DataFrame(v).iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe, using list comporehension
    comps = pd.DataFrame(list(zip(v_1, features_list)),
                         columns=['weights', 'features'])

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights', ascending=False).head(n_weights)

    # display using seaborn
    ax=plt.subplots(figsize=(10,6))
    ax=sns.barplot(data=sorted_weight_data,
                   x="weights",
                   y="features",
                   palette="Blues_d")
    ax.set_title("PCA Component Makeup, Component #" + str(component_num))
    plt.show()


def box_plot(df,categories,nr_cols):
    import matplotlib.pyplot as plt
    import seaborn as sns
    nr_rows = len(categories)//nr_cols+1
    li_cat_feats = list(categories)
    fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))
    for r in range(0,nr_rows):
        for c in range(0,nr_cols):
            i = r*nr_cols+c
            if i < len(li_cat_feats):
                sns.boxplot(x=li_cat_feats[i], data=df, ax = axs[r][c])
    plt.tight_layout()
    plt.show()

def count_plot(df,categories,nr_cols):
    import matplotlib.pyplot as plt
    import seaborn as sns
    nr_rows = len(categories)//nr_cols+1
    li_cat_feats = list(categories)
    fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))
    for r in range(0,nr_rows):
        for c in range(0,nr_cols):
            i = r*nr_cols+c
            if i < len(li_cat_feats):
                sns.countplot(x=li_cat_feats[i], data=df, ax = axs[r][c])
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test,y_prediction):
    import itertools
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score,recall_score,precision_score,f1_score,confusion_matrix,accuracy_score
    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_prediction)
    labels = ['0', '1']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    fmt = '.1f'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")
    plt.grid('off')
    plt.show()

# create dimensionality-reduced data
def create_transformed_df(train_pca, df_scaled, n_top_components):
    ''' Return a dataframe of data points with component features.
        The dataframe should be indexed by State-County and contain component values.
        :param train_pca: A list of pca training data, returned by a PCA model.
        :param counties_scaled: A dataframe of normalized, original features.
        :param n_top_components: An integer, the number of top components to use.
        :return: A dataframe, indexed by State-County, with n_top_component values as columns.
     '''
    # create new dataframe to add data to
    df_transformed=pd.DataFrame()

    # for each of our new, transformed data points
    # append the component values to the dataframe
    for data in train_pca:
        # get component values for each data point
        components=data.label['projection'].float32_tensor.values
        df_transformed=df_transformed.append([list(components)])

    # index by county, just like counties_scaled
    df_transformed.index=df_scaled.index

    # keep only the top n components
    start_idx = N_COMPONENTS - n_top_components
    df_transformed = df_transformed.iloc[:,start_idx:]

    # reverse columns, component order
    return df_transformed.iloc[:, ::-1]

def variance_threshold_selector(df,target,threshold):
    selector = VarianceThreshold(threshold)
    X = df.drop(target,axis=1)
    selector.fit(X)
    feat_ix_keep = selector.get_support(indices=True)
    orig_feat_ix = np.arange(X.columns.size)
    feat_ix_delete = np.delete(orig_feat_ix, feat_ix_keep)

    print('Se eliminarion {} columnas del dataset.'.format(len(X.columns[feat_ix_delete])))
    print('Columnas eliminadas : ')
    print('\n')
    print(df.columns[feat_ix_delete])
    df.drop(X.columns[feat_ix_delete],axis=1,inplace=True)

# Funcion para eliminar outliers a partir del rango interquartile
def drop_outliers(df, field_name):
    IQR = (df[field_name].quantile(.75) - df[field_name].quantile(.25))
    K=3#=1.5
    df.drop(df[df[field_name] > df[field_name].quantile(.75)+(IQR*K)].index, inplace=True)
    df.drop(df[df[field_name] < df[field_name].quantile(.25)-(IQR*K)].index, inplace=True)

def elbow_method_computing(df,min_k,max_k,step):
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    import time
    res_clusters = pd.DataFrame({'k': np.arange(min_k, max_k, step)}, columns=['k'])
    res_clusters['wcss'] = 0

    for ind in res_clusters.index:
        tic = time.time()
        val_k = res_clusters.at[ind,'k']
        clusterer = KMeans(n_clusters = val_k, random_state = 42, n_jobs=-1)
        cluster_labels = clusterer.fit_predict(df)

        wcss = clusterer.inertia_
        res_clusters.at[ind,'wcss'] = wcss
        toc = time.time()
        print(val_k,'Time Training :{:.2f} minutes '.format((toc-tic)/60))

    return res_clusters


def silhouette_kmeans(df,clusters,step):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import datetime
    for K in range(2,clusters,step):
        cluster_kmeans = KMeans(n_clusters=K, random_state=np.random.randint(42), n_jobs=-1)
        cluster_kmeans.fit(df)
        y_kmeans = cluster_kmeans.predict(df)
        centers_=cluster_kmeans.cluster_centers_
        silhouette_avg = silhouette_score(df, y_kmeans)
        print('Number of K = {}, The average silhouette_score is :{}'.format(K,silhouette_avg),datetime.datetime.now())
