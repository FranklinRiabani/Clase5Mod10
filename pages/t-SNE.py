import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

# agregar configuracion de pagina

st.set_page_config(page_title='t-SNE', layout='wide')

st.title('t-SNE del Modulo 10')

st.sidebar.title('Menu de opciones')

#lista de opciones
opciones = ['Cargar datos','t-SNE']

#seleccionar una opcion
opcion = st.sidebar.selectbox('Seleccione una opcion',opciones)

@st._cache_data
def cargar_datos(archivo):
    if archivo:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith('.xlsx'):
            df = pd.read_excel(archivo)
        else:
            raise ValueError("Formato de archivo no soportado. Solo se aceptan archivos CSV y XLSX.")
        return df
    else:
        return None

if opcion == 'Cargar datos':
    st.sidebar.subheader('Cargar datos')
    archivo=st.sidebar.file_uploader('Seleccione un archivo CSV o XLSX',type=['csv','xlsx'])
    if archivo:
        df = cargar_datos(archivo)
        st.session_state.df = df
        st.info('Datos cargados correctamente')
    else:
        st.write('No hay datos para mostrar')
elif opcion == 't-SNE':
    st.title('t-SNE')
    if 'df' not in st.session_state:
        st.warning('No hay datos cargados')
    else:
        df=st.session_state.df
        st.write('El archivo contiene {} filas y {} columnas'.format(df.shape[0],df.shape[1]))

        n_components = st.sidebar.slider('Numero de componentes',2,10,2)

        perplexity = st.sidebar.slider('Perplexity',5,50,30)

        n_sample = st.sidebar.slider('Numero de muestras',1000,70000,5000)

        x=np.asanyarray(df.drop(columns=['class']))[:n_sample,:]
        y=np.asanyarray(df['class'])[:n_sample].ravel()

        tsne = TSNE(n_components=n_components,perplexity=perplexity)

        x2=tsne.fit_transform(x)

        # graficar
       

        if n_components==2:
            fig= plt.figure(figsize=(6,6))
            sns.scatterplot(x=x2[:,0],y=x2[:,1],hue=y,palette='viridis')
            st.pyplot(fig)
        else:
            fig_3d=plt.figure(figsize=(6,6))
            ax=fig_3d.add_subplot(111,projection='3d')
            ax.scatter(x2[:,0],x2[:,1],x2[:,2],c=y,cmap='viridis')
            st.pyplot(fig_3d)




