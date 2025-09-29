# =============================================================================
# IMPORTATION DES MODULES UTILISÉS
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
from datetime import datetime
import os
import sys
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import base64
import tempfile
import ast
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# Import des méthodes de clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Nouveaux imports pour les datasets sklearn et la régression
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import scipy.optimize as opt

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Analyse Avancée des Résultats d'Entraînement IA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# FONCTIONS UTILITAIRES AMÉLIORÉES
# =============================================================================
# Optimisations pour la production    
@st.cache_data(show_spinner=False)
def configure_streamlit_cloud():
    """Configuration optimisée pour Streamlit Cloud"""
    # Désactiver les warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Optimisations pandas
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    
    # Configurer matplotlib pour le backend non interactif
    plt.switch_backend('Agg')
    
    # Vérifier l'environnement Cloud
    if 'STREAMLIT_CLOUD' in os.environ:
        st.info("🌐 Application exécutée sur Streamlit Cloud")

def detect_separator(content):
    """Détecte automatiquement le séparateur utilisé dans le fichier"""
    # Analyser les premières lignes pour détecter le séparateur
    lines = content.split('\n')[:10]
    
    # Compter les occurrences de chaque séparateur potentiel
    separators = [',', ';', '\t', '|', ' ']
    separator_counts = {sep: 0 for sep in separators}
    
    for line in lines:
        if line.strip():
            for sep in separators:
                separator_counts[sep] += line.count(sep)
    
    # Retourner le séparateur le plus fréquent
    best_separator = max(separator_counts, key=separator_counts.get)
    
    # Si aucun séparateur n'est détecté, essayer avec les expressions régulières
    if separator_counts[best_separator] == 0:
        # Essayer avec les espaces multiples
        if any(re.search(r'\s{2,}', line) for line in lines if line.strip()):
            return r'\s+'
    
    return best_separator

def load_data(uploaded_file, separator=None):
    """Charge un fichier de données avec gestion robuste des erreurs"""
    try:
        if uploaded_file.name.endswith('.csv'):
            if separator:
                return pd.read_csv(uploaded_file, sep=separator)
            else:
                # Détection automatique du séparateur
                content = uploaded_file.getvalue().decode('utf-8')
                detected_separator = detect_separator(content)
                st.info(f"🔍 Séparateur détecté: '{detected_separator}'")
                return pd.read_csv(uploaded_file, sep=detected_separator)
                
        elif uploaded_file.name.endswith('.txt'):
            if separator:
                return pd.read_csv(uploaded_file, sep=separator)
            else:
                # Détection automatique du séparateur
                content = uploaded_file.getvalue().decode('utf-8')
                detected_separator = detect_separator(content)
                st.info(f"🔍 Séparateur détecté: '{detected_separator}'")
                return pd.read_csv(uploaded_file, sep=detected_separator, engine='python')
                
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            st.error(f"Format de fichier non supporté: {uploaded_file.name}")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement de {uploaded_file.name}: {e}")
        return None

def load_sklearn_dataset(dataset_name):
    """Charge un dataset de sklearn avec une gestion améliorée"""
    dataset_map = {
        'Iris': datasets.load_iris,
        'Diabète': datasets.load_diabetes,
        'Digits': datasets.load_digits,
        'Linnerud': datasets.load_linnerud,
        'Wine': datasets.load_wine,
        'Breast Cancer': datasets.load_breast_cancer,
        'California Housing': datasets.fetch_california_housing,
    }
    
    try:
        if dataset_name in dataset_map:
            data = dataset_map[dataset_name]()
            
            # Gestion différenciée selon le type de dataset
            if hasattr(data, 'frame') and data.frame is not None:
                # Cas des datasets avec format DataFrame intégré
                df = data.frame
            elif hasattr(data, 'data') and hasattr(data, 'feature_names'):
                # Cas standard avec data et feature_names
                df = pd.DataFrame(data.data, columns=data.feature_names)
                
                # Ajouter la target si elle existe
                if hasattr(data, 'target'):
                    if hasattr(data, 'target_names'):
                        # Pour les datasets de classification, mapper les targets aux noms
                        target_names = data.target_names
                        target_values = data.target
                        
                        # Vérifier si c'est une classification multi-classes
                        if len(target_names) > 0:
                            # Créer une colonne avec les noms des classes
                            if len(target_names) == len(np.unique(target_values)):
                                df['target'] = target_values
                                df['target_name'] = [target_names[i] for i in target_values]
                            else:
                                # Cas où les target_names ne correspondent pas directement aux valeurs
                                df['target'] = target_values
                        else:
                            df['target'] = target_values
                    else:
                        # Pas de noms de classes, juste les valeurs
                        df['target'] = data.target
            else:
                st.error(f"Format de dataset non supporté: {dataset_name}")
                return None
            
            # Nettoyer les noms de colonnes
            df.columns = [col.replace(' ', '_').replace('<', '').replace('>', '').lower() for col in df.columns]
            
            # S'assurer qu'il n'y a pas de colonnes dupliquées
            df = df.loc[:, ~df.columns.duplicated()]
            
            return df
        else:
            st.error(f"Dataset {dataset_name} non trouvé")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du dataset {dataset_name}: {str(e)}")
        import traceback
        st.error(f"Détails de l'erreur: {traceback.format_exc()}")
        return None

def detect_column_types(df, max_categories=20):
    """Détecte automatiquement les types de colonnes avec une logique améliorée"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = []
    
    # Colonnes avec peu de valeurs uniques (même si numériques)
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count <= max_categories and unique_count > 0:
            categorical_cols.append(col)
    
    # Retirer des numériques celles qui sont catégorielles
    numeric_cols = [col for col in numeric_cols if col not in categorical_cols]
    
    # Colonnes texte (beaucoup de valeurs manquantes)
    text_cols = [col for col in df.columns if col not in numeric_cols + categorical_cols]
    
    return numeric_cols, categorical_cols, text_cols

def calculate_missing_rows(df, missing_threshold=60):
    """Calcule les lignes avec trop de valeurs manquantes"""
    if df.empty:
        return 0, 0.0
    
    # Calculer le pourcentage de valeurs manquantes par ligne
    missing_per_row = df.isnull().mean(axis=1) * 100
    # Compter les lignes avec plus de missing_threshold % de valeurs manquantes
    high_missing_rows = (missing_per_row > missing_threshold).sum()
    percentage = (high_missing_rows / len(df)) * 100 if len(df) > 0 else 0
    
    return high_missing_rows, percentage

def identify_ignored_jobs(df_list, missing_thresholds=None):
    """Identifie les jobs à ignorer avec des critères améliorés"""
    if missing_thresholds is None:
        missing_thresholds = [0.7, 0.7, 0.6]  # Seuils pour chaque fichier et global
    
    ignored_jobs = {'reason': {}, 'count': 0, 'total_jobs': len(df_list[0]) if df_list else 0}
    
    if not df_list:
        return ignored_jobs
    
    for idx in range(len(df_list[0])):
        reasons = []
        
        # Critère 1: Valeurs manquantes dans au moins 2 fichiers
        high_missing_count = 0
        for i, df in enumerate(df_list):
            if i < len(missing_thresholds):
                missing_ratio = df.iloc[idx].isnull().mean()
                if missing_ratio >= missing_thresholds[i]:
                    high_missing_count += 1
        
        if high_missing_count >= 2:
            reasons.append(f"Trop de valeurs manquantes dans {high_missing_count} fichiers")
        
        # Critère 2: Valeurs aberrantes dans les métriques principales
        for df in df_list:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:3]:  # Vérifier les 3 premières colonnes numériques
                if col in df.columns and not pd.isna(df.iloc[idx][col]):
                    value = df.iloc[idx][col]
                    col_data = df[col].dropna()
                    if len(col_data) > 10:
                        Q1, Q3 = col_data.quantile([0.25, 0.75])
                        IQR = Q3 - Q1
                        if value < (Q1 - 3 * IQR) or value > (Q3 + 3 * IQR):
                            reasons.append(f"Valeur aberrante dans {col}")
                            break
        
        if reasons:
            ignored_jobs['reason'][idx] = reasons
            ignored_jobs['count'] += 1
    
    return ignored_jobs

def apply_basic_filters(df, filters):
    """Applique les filtres de base aux données (non destructif)"""
    filtered_df = df.copy()
    
    for col, filter_settings in filters.items():
        if col not in filtered_df.columns:
            continue
            
        filter_type, values = filter_settings
        
        if filter_type == 'range' and pd.api.types.is_numeric_dtype(filtered_df[col]):
            min_val, max_val = values
            if not pd.isna(min_val) and not pd.isna(max_val):
                filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
        elif filter_type == 'categories':
            if values:  # Ne filtrer que si des valeurs sont sélectionnées
                filtered_df = filtered_df[filtered_df[col].isin(values)]
        elif filter_type == 'condition':
            # Filtre conditionnel simple
            try:
                mask = filtered_df.eval(values)
                filtered_df = filtered_df[mask]
            except:
                st.warning(f"Impossible d'appliquer la condition: {values}")
    
    return filtered_df

def apply_advanced_filters(df, advanced_filters, column_mapping):
    """Applique des filtres avancés avec mapping des colonnes"""
    filtered_df = df.copy()
    
    for filter_name, filter_info in advanced_filters.items():
        filter_expr = filter_info['expression']
        column_map = filter_info.get('column_mapping', {})
        filter_type = filter_info.get('type', 'custom')  # 'predefined' ou 'custom'
        
        try:
            # Remplacer les noms de colonnes génériques par les vraies colonnes
            replaced_expr = filter_expr
            for generic_col, actual_col in column_map.items():
                if actual_col:  # Vérifier qu'une colonne a été sélectionnée
                    replaced_expr = replaced_expr.replace(generic_col, actual_col)
            
            # Pour les filtres prédéfinis, remplacer aussi les valeurs de seuil
            if filter_type == 'predefined' and 'threshold' in filter_info:
                threshold_value = filter_info['threshold']
                replaced_expr = replaced_expr.replace('threshold', str(threshold_value))
            
            # Préparer l'expression pour eval
            safe_expr = replaced_expr.replace('^', '**')  # Convertir la notation puissance
            
            # Créer un masque avec l'expression
            mask = filtered_df.eval(safe_expr)
            filtered_df = filtered_df[mask]
            
            st.success(f"Filtre '{filter_name}' appliqué: {mask.sum()} lignes conservées")
        except Exception as e:
            st.error(f"Erreur dans le filtre '{filter_name}': {e}")
    
    return filtered_df

def calculate_brice_correlation(x, y):
    """
    Calcule les corrélations Brice (non-centrée) et Brice1 (centrée).
    L'implémentation de Brice est corrigée pour utiliser les sommes
    conformément à la formule du cosinus entre vecteurs.
    """
    try:
        # Convertir en arrays numpy
        x = np.asarray(x).astype(float)
        y = np.asarray(y).astype(float)
        
        # Masque des positions où x et y ne sont pas NaN
        mask = (~np.isnan(x)) & (~np.isnan(y))
        
        # Si moins de 4 points valides, corrélation non significative
        if mask.sum() < 4:
            return {'brice': np.nan, 'brice1': np.nan}
        
        # Filtrage effectif : ne garder que les paires valides
        x, y = x[mask], y[mask]
        
        # --- 1. Variations (Dérivées Discrètes) ---
        x1 = x[1:] - x[:-1]
        y1 = y[1:] - y[:-1]
        
        # Filtrer les NaN créés par les différences
        mask_diff = (~np.isnan(x1)) & (~np.isnan(y1))
        x1, y1 = x1[mask_diff], y1[mask_diff]

        # --- 2. Corrélation Brice (Non-Centrée/Cosinus) ---
        # Formule: Sum(x1*y1) / [sqrt(Sum(x1^2) * Sum(y1^2))]
        
        sum_xy = np.nansum(x1 * y1)
        sum_x2 = np.nansum(x1 * x1)
        sum_y2 = np.nansum(y1 * y1)
        
        denominator = np.sqrt(sum_x2 * sum_y2)
        
        brice_corr = sum_xy / denominator if denominator != 0 else np.nan
        
        # --- 3. Corrélation Brice1 (Centrée/Pearson Classique) ---
        # Formule: Corr(x1, y1)
        
        x1_centered = x1 - np.nanmean(x1)
        y1_centered = y1 - np.nanmean(y1)
        
        # La covariance des variations
        covariance = np.nanmean(x1_centered * y1_centered)
        
        # L'écart-type des variations centrées (sqrt de la variance)
        std_x1_centered = np.sqrt(np.nanmean(x1_centered * x1_centered))
        std_y1_centered = np.sqrt(np.nanmean(y1_centered * y1_centered))
        
        std_product = std_x1_centered * std_y1_centered
        
        brice1_corr = covariance / std_product if std_product != 0 else np.nan
        
        return {'brice': brice_corr, 'brice1': brice1_corr}
    
    except Exception as e:
        # Gérer toute erreur imprévue (comme une entrée non valide)
        print(f"Erreur lors du calcul: {e}")
        return {'brice': np.nan, 'brice1': np.nan}


def create_custom_plot(data, x_col, y_col=None, z_col=None, plot_type="histogram", 
                      color_col=None, facet_col=None, theme_settings=None,
                      custom_colors=None, fig_size=None, font_sizes=None,
                      nbins=30, line_width=2, density=False, trendline=None,
                      heatmap_columns=None, max_heatmap_cols=15, dimensions=2):
    """Crée des graphiques personnalisés avec une grande flexibilité (2D et 3D)"""
    
    if theme_settings is None:
        theme_settings = {}
    if custom_colors is None:
        custom_colors = px.colors.qualitative.Plotly
    if fig_size is None:
        fig_size = {'width': 800, 'height': 600}
    if font_sizes is None:
        font_sizes = {'title': 16, 'axis': 14, 'legend': 12}
    
    try:
        # Gestion des graphiques 3D
        if dimensions == 3 and plot_type in ["scatter", "line", "density"]:
            if plot_type == "scatter" and y_col and z_col:
                fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col, color=color_col,
                                  title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}",
                                  color_discrete_sequence=custom_colors)
            elif plot_type == "line" and y_col and z_col:
                fig = px.line_3d(data, x=x_col, y=y_col, z=z_col, color=color_col,
                               title=f"3D Line: {x_col} vs {y_col} vs {z_col}",
                               color_discrete_sequence=custom_colors)
            elif plot_type == "density" and y_col and z_col:
                # Pour la densité 3D, on utilise un scatter 3D avec taille proportionnelle à la densité
                fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col, color=color_col,
                                  title=f"3D Density: {x_col} vs {y_col} vs {z_col}",
                                  color_discrete_sequence=custom_colors)
            else:
                # Fallback à 2D si les colonnes nécessaires ne sont pas fournies
                dimensions = 2
        
        # Si ce n'est pas un graphique 3D ou si le fallback a été appliqué
        if dimensions == 2:
            # SECTION SPÉCIALISÉE POUR LES HEATMAPS
            if plot_type in ["heatmap", "heatmap_brice", "heatmap_brice1"]:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                
                # Filtrer les colonnes sélectionnées pour ne garder que les numériques
                if heatmap_columns:
                    valid_cols = [col for col in heatmap_columns if col in numeric_cols]
                    if not valid_cols:
                        st.warning("Aucune colonne numérique valide sélectionnée. Utilisation des colonnes par défaut.")
                        if len(numeric_cols) > max_heatmap_cols:
                            numeric_cols = numeric_cols[:max_heatmap_cols]
                    else:
                        numeric_cols = valid_cols
                else:
                    # Par défaut, utiliser les premières colonnes numériques
                    if len(numeric_cols) > max_heatmap_cols:
                        numeric_cols = numeric_cols[:max_heatmap_cols]
                
                if len(numeric_cols) < 2:
                    st.error("Il faut au moins 2 colonnes numériques pour créer une heatmap")
                    return None
                
                # Nettoyer les données : convertir en numérique et supprimer les NaN
                try:
                    clean_data = data[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
                    
                    if len(clean_data) < 2:
                        st.error("Pas assez de données valides après nettoyage pour calculer les corrélations")
                        return None
                    
                    if plot_type == "heatmap":
                        # Heatmap standard (corrélation de Pearson)
                        corr_matrix = clean_data.corr()
                        fig = px.imshow(corr_matrix, 
                                       title="Matrice de corrélation (Pearson)",
                                       color_continuous_scale='RdBu_r', 
                                       aspect="auto",
                                       text_auto=True,
                                       zmin=-1, zmax=1)  # Forcer l'échelle entre -1 et 1
                        fig.update_xaxes(side="top")
                    
                    elif plot_type == "heatmap_brice":
                        # Heatmap avec corrélation Brice
                        n = len(numeric_cols)
                        brice_matrix = np.zeros((n, n))
                        
                        for i in range(n):
                            brice_matrix[i, i] = 1.0
                            for j in range(i+1, n):
                                corr_result = calculate_brice_correlation(
                                    clean_data[numeric_cols[i]], clean_data[numeric_cols[j]]
                                )
                                brice_matrix[i, j] = brice_matrix[j, i] = corr_result['brice']
                        
                        fig = px.imshow(brice_matrix, 
                                       x=numeric_cols, y=numeric_cols,
                                       title="Matrice de corrélation Brice",
                                       color_continuous_scale='RdBu_r', 
                                       aspect="auto",
                                       text_auto=True,
                                       zmin=-1, zmax=1)
                    
                    elif plot_type == "heatmap_brice1":
                        # Heatmap avec corrélation Brice1
                        n = len(numeric_cols)
                        brice1_matrix = np.zeros((n, n))
                        
                        for i in range(n):
                            brice1_matrix[i, i] = 1.0
                            for j in range( i+1, n):
                                    corr_result = calculate_brice_correlation(
                                        clean_data[numeric_cols[i]], clean_data[numeric_cols[j]]
                                    )
                                    brice1_matrix[i, j] = brice1_matrix[j, i] = corr_result['brice1']
                        
                        fig = px.imshow(brice1_matrix, 
                                       x=numeric_cols, y=numeric_cols,
                                       title="Matrice de corrélation Brice1",
                                       color_continuous_scale='RdBu_r', 
                                       aspect="auto",
                                       text_auto=True,
                                       zmin=-1, zmax=1)
                    
                    # Application des paramètres de thème pour les heatmaps
                    fig.update_layout(
                        font_family=theme_settings.get('font_family', 'Arial'),
                        font_size=theme_settings.get('font_size', 12),
                        template=theme_settings.get('color_theme', 'plotly_white'),
                        width=fig_size['width'],
                        height=fig_size['height'],
                        title_font_size=font_sizes['title'],
                        xaxis_title_font_size=font_sizes['axis'],
                        yaxis_title_font_size=font_sizes['axis'],
                    )
                    
                    return fig
                    
                except Exception as e:
                    st.error(f"Erreur lors du calcul de la heatmap: {e}")
                    return None
            
            # SECTION POUR LES AUTRES TYPES DE GRAPHIQUES 2D
            # Gérer le cas où x_col et y_col sont identiques
            if y_col == x_col:
                # Pour les graphiques qui nécessitent deux variables différentes
                if plot_type in ["scatter", "line", "density"]:
                    # Créer un graphique de distribution simple à la place
                    fig = px.histogram(data, x=x_col, title=f"Distribution de {x_col}",
                                      nbins=nbins, color_discrete_sequence=custom_colors)
                    if density:
                        # Ajouter une courbe de densité
                        fig.add_trace(go.Scatter(
                            x=np.linspace(data[x_col].min(), data[x_col].max(), 100),
                            y=np.histogram(data[x_col].dropna(), bins=nbins, density=True)[0],
                            mode='lines', name='Densité', line=dict(color='red', width=2)
                        ))
                else:
                    # Pour les autres types, créer le graphique normal
                    if plot_type == "histogram":
                        fig = px.histogram(data, x=x_col, color=color_col, facet_col=facet_col,
                                          title=f"Distribution de {x_col}", nbins=nbins,
                                          color_discrete_sequence=custom_colors)
                        if density:
                            # Ajouter une courbe de densité
                            fig.add_trace(go.Scatter(
                                x=np.linspace(data[x_col].min(), data[x_col].max(), 100),
                                y=np.histogram(data[x_col].dropna(), bins=nbins, density=True)[0],
                                mode='lines', name='Densité', line=dict(color='red', width=2)
                            ))
                    
                    elif plot_type == "box":
                        fig = px.box(data, y=x_col, color=color_col, title=f"Boxplot de {x_col}",
                                    color_discrete_sequence=custom_colors)
                    
                    elif plot_type == "violin":
                        fig = px.violin(data, y=x_col, color=color_col, box=True,
                                       title=f"Distribution de {x_col} (violon)",
                                       color_discrete_sequence=custom_colors)
                    
                    elif plot_type == "bar":
                        # Diagramme en barres pour une seule variable
                        value_counts = data[x_col].value_counts().reset_index()
                        value_counts.columns = ['Category', 'Count']
                        fig = px.bar(value_counts, x='Category', y='Count', 
                                    title=f"Distribution de {x_col}",
                                    color_discrete_sequence=custom_colors)
                    
                    elif plot_type == "pie":
                        value_counts = data[x_col].value_counts()
                        fig = px.pie(values=value_counts.values, names=value_counts.index, 
                                    title=f"Distribution de {x_col}")
                    
                    else:
                        fig = px.histogram(data, x=x_col, title=f"Distribution de {x_col}",
                                          nbins=nbins, color_discrete_sequence=custom_colors)
            else:
                # Cas normal où x_col et y_col sont différentes
                if plot_type == "histogram":
                    fig = px.histogram(data, x=x_col, color=color_col, facet_col=facet_col,
                                      title=f"Distribution de {x_col}", nbins=nbins,
                                      color_discrete_sequence=custom_colors)
                    if density:
                        # Ajouter une courbe de densité
                        fig.add_trace(go.Scatter(
                            x=np.linspace(data[x_col].min(), data[x_col].max(), 100),
                            y=np.histogram(data[x_col].dropna(), bins=nbins, density=True)[0],
                            mode='lines', name='Densité', line=dict(color='red', width=2)
                        ))
                
                elif plot_type == "scatter" and y_col:
                    fig = px.scatter(data, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                                   title=f"{y_col} vs {x_col}", trendline=trendline,
                                   color_discrete_sequence=custom_colors)
                
                elif plot_type == "line" and y_col:
                    fig = px.line(data, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                                 title=f"{y_col} en fonction de {x_col}",
                                 color_discrete_sequence=custom_colors)
                
                elif plot_type == "box":
                    if y_col:
                        fig = px.box(data, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                                    title=f"Boxplot de {y_col} par {x_col}",
                                    color_discrete_sequence=custom_colors)
                    else:
                        fig = px.box(data, y=x_col, color=color_col, facet_col=facet_col,
                                    title=f"Boxplot de {x_col}",
                                    color_discrete_sequence=custom_colors)
                
                elif plot_type == "violin":
                    if y_col:
                        fig = px.violin(data, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                                       box=True, title=f"Distribution de {y_col} par {x_col} (violon)",
                                       color_discrete_sequence=custom_colors)
                    else:
                        fig = px.violin(data, y=x_col, color=color_col, facet_col=facet_col,
                                       box=True, title=f"Distribution de {x_col} (violon)",
                                       color_discrete_sequence=custom_colors)
                
                elif plot_type == "density" and y_col:
                    # CORRECTION: Utiliser density_contour pour les courbes de densité 2D
                    fig = px.density_contour(data, x=x_col, y=y_col, color=color_col,
                                            title=f"Densité de {x_col} vs {y_col}",
                                            color_discrete_sequence=custom_colors)
                    # Ajouter les points pour une meilleure visualisation
                    fig.add_trace(go.Scatter(x=data[x_col], y=data[y_col], 
                                           mode='markers', marker=dict(size=3, opacity=0.3),
                                           name='Points', showlegend=False))
                
                elif plot_type == "bar":
                    if y_col:
                        # Diagramme en barres avec deux variables
                        fig = px.bar(data, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                                    title=f"{y_col} par {x_col}",
                                    color_discrete_sequence=custom_colors)
                    else:
                        # Diagramme en barres pour une seule variable
                        value_counts = data[x_col].value_counts().reset_index()
                        value_counts.columns = ['Category', 'Count']
                        fig = px.bar(value_counts, x='Category', y='Count', 
                                    title=f"Distribution de {x_col}",
                                    color_discrete_sequence=custom_colors)
                
                elif plot_type == "pie":
                    value_counts = data[x_col].value_counts()
                    fig = px.pie(values=value_counts.values, names=value_counts.index, 
                                title=f"Distribution de {x_col}")
                
                elif plot_type == "pairplot":
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 5:
                        numeric_cols = numeric_cols[:5]
                    fig = px.scatter_matrix(data[numeric_cols], title="Pairplot des variables",
                                           color_discrete_sequence=custom_colors)
                
                else:
                    # Cas par défaut : histogramme
                    fig = px.histogram(data, x=x_col, title=f"Distribution de {x_col}",
                                      nbins=nbins, color_discrete_sequence=custom_colors)
        
        # Application des paramètres de thème
        if fig:
            fig.update_layout(
                font_family=theme_settings.get('font_family', 'Arial'),
                font_size=theme_settings.get('font_size', 12),
                template=theme_settings.get('color_theme', 'plotly_white'),
                width=fig_size['width'],
                height=fig_size['height'],
                title_font_size=font_sizes['title'],
                xaxis_title_font_size=font_sizes['axis'],
                yaxis_title_font_size=font_sizes['axis'],
                legend_font_size=font_sizes['legend']
            )
        
        return fig
    
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique: {e}")
        return None

def analyze_sampling_data_improved(df, interval_prefix="Interval_", 
                                 stat_prefixes=None, num_intervals_prefix="Numb_intervals_"):
    """Analyse améliorée des données d'échantillonnage"""
    
    if stat_prefixes is None:
        stat_prefixes = ["Ech_count_", "Ech_mean_", "Ech_std_", "Ech_min_", "Ech_max_"]
    
    analysis_results = {}
    
    # Trouver tous les suffixes uniques
    all_interval_cols = [col for col in df.columns if col.startswith(interval_prefix)]
    suffixes = set()
    
    for col in all_interval_cols:
        # Extraire le suffixe (ex: "_tr_te" de "Interval_0_tr_te")
        parts = col.replace(interval_prefix, "").split("_")
        if len(parts) > 1:
            suffix = "_" + "_".join(parts[1:])
            suffixes.add(suffix)
    
    for suffix in suffixes:
        suffix_data = {}
        
        # Trouver le nombre d'intervalles pour ce suffixe
        num_intervals_col = f"{num_intervals_prefix}{suffix}"
        if num_intervals_col in df.columns:
            num_intervals = int(df[num_intervals_col].iloc[0]) if not pd.isna(df[num_intervals_col].iloc[0]) else 0
        else:
            # Estimer le nombre d'intervalles à partir des colonnes disponibles
            matching_cols = [col for col in df.columns if f"_{suffix}" in col]
            if matching_cols:
                interval_cols = [col for col in matching_cols if col.startswith(interval_prefix)]
                num_intervals = len(interval_cols)
            else:
                num_intervals = 0
        
        # Collecter les données pour chaque statistique
        for stat_prefix in stat_prefixes:
            stat_col = f"{stat_prefix}{suffix}"
            if stat_col in df.columns:
                # Les données d'échantillonnage sont stockées dans une seule cellule
                cell_value = df[stat_col].iloc[0]
                
                if isinstance(cell_value, str) and '[' in cell_value:
                    # Convertir la chaîne en liste
                    try:
                        data_list = ast.literal_eval(cell_value)
                        if len(data_list) == num_intervals:
                            suffix_data[stat_prefix] = data_list
                    except:
                        # Si l'évaluation échoue, essayer de parser manuellement
                        try:
                            data_list = [float(x.strip()) for x in cell_value.strip('[]').split(',')]
                            if len(data_list) == num_intervals:
                                suffix_data[stat_prefix] = data_list
                        except:
                            st.warning(f"Impossible de parser les données pour {stat_col}")
                elif isinstance(cell_value, (int, float)):
                    # Donnée unique - créer une liste avec cette valeur répétée
                    suffix_data[stat_prefix] = [cell_value] * num_intervals
        
        if suffix_data:
            analysis_results[suffix] = {
                'data': suffix_data,
                'num_intervals': num_intervals,
                'suffix': suffix
            }
    
    return analysis_results

def create_sampling_visualization(analysis_results, selected_suffixes, 
                                selected_stats, visualization_type="line",
                                theme_settings=None, fig_size=None):
    """Crée des visualisations avancées pour les données d'échantillonnage"""
    
    if not selected_suffixes or not selected_stats:
        return None
    
    # Créer un subplot avec une ligne par statistique
    fig = make_subplots(
        rows=len(selected_stats), cols=1,
        subplot_titles=[f"Évolution de {stat}" for stat in selected_stats],
        vertical_spacing=0.05
    )
    
    colors = px.colors.qualitative.Plotly
    
    for stat_idx, stat in enumerate(selected_stats, 1):
        for suffix_idx, suffix in enumerate(selected_suffixes):
            if suffix in analysis_results and stat in analysis_results[suffix]['data']:
                data = analysis_results[suffix]['data'][stat]
                x_values = list(range(len(data)))
                
                if visualization_type == "line":
                    fig.add_trace(
                        go.Scatter(x=x_values, y=data, mode='lines+markers',
                                 name=f"{suffix} - {stat}",
                                 line=dict(color=colors[suffix_idx % len(colors)])),
                        row=stat_idx, col=1
                    )
                elif visualization_type == "bar":
                    fig.add_trace(
                        go.Bar(x=x_values, y=data, name=f"{suffix} - {stat}",
                              marker_color=colors[suffix_idx % len(colors)]),
                        row=stat_idx, col=1
                    )
        
        # Configurer l'axe y pour ce subplot
        fig.update_yaxes(title_text=stat, row=stat_idx, col=1)
    
    # Configuration finale
    fig.update_layout(
        height=300 * len(selected_stats),
        title_text="Analyse d'échantillonnage - Évolution des statistiques",
        showlegend=True,
        font=dict(
            family=theme_settings.get('font_family', 'Arial'),
            size=theme_settings.get('font_size', 12)
        )
    )
    
    if fig_size:
        fig.update_layout(width=fig_size.get('width', 800), 
                         height=fig_size.get('height', 600))
    
    return fig

def save_plot_as_pdf(fig, filename="graphique.pdf"):
    """Sauvegarde un graphique Plotly au format PDF amélioré"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name, scale=2, width=1000, height=800)
            
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            width, height = letter
            
            img = ImageReader(tmpfile.name)
            img_width, img_height = img.getSize()
            aspect = img_height / float(img_width)
            
            display_width = width * 0.9
            display_height = display_width * aspect
            
            if display_height > height * 0.9:
                display_height = height * 0.9
                display_width = display_height / aspect
            
            x = (width - display_width) / 2
            y = (height - display_height) / 2
            
            c.drawImage(tmpfile.name, x, y, display_width, display_height)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "Analyse des Résultats d'Entraînement IA")
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 70, f"Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            c.showPage()
            c.save()
            
            os.unlink(tmpfile.name)
            return pdf_buffer.getvalue()
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde du PDF: {e}")
        return None

def detect_outliers_iqr(df, column):
    """Détecte les outliers avec la méthode IQR"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def create_time_series_analysis(df, date_column, value_column):
    """Crée une analyse de séries temporelles"""
    df_sorted = df.sort_values(date_column)
    fig = px.line(df_sorted, x=date_column, y=value_column, 
                  title=f"Série temporelle de {value_column}")
    
    # Ajouter une moyenne mobile
    df_sorted['moving_avg'] = df_sorted[value_column].rolling(window=7).mean()
    fig.add_trace(go.Scatter(x=df_sorted[date_column], y=df_sorted['moving_avg'],
                            mode='lines', name='Moyenne mobile (7)',
                            line=dict(color='red', dash='dash')))
    
    return fig

# =============================================================================
# FONCTIONS AMÉLIORÉES POUR LE CLUSTERING AVANCÉ
# =============================================================================

def apply_clustering_method(data, method, n_clusters=3, **kwargs):
    """Applique différentes méthodes de clustering"""
    
    # Standardiser les données
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == "DBSCAN":
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "Agglomerative":
        linkage = kwargs.get('linkage', 'ward')
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    elif method == "Spectral":
        model = SpectralClustering(n_clusters=n_clusters, random_state=42)
    elif method == "Gaussian Mixture":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    else:
        raise ValueError(f"Méthode de clustering non supportée: {method}")
    
    # Appliquer le clustering (CORRECTION POUR GAUSSIAN MIXTURE)
    if method in ["DBSCAN", "Agglomerative"]:
        clusters = model.fit_predict(scaled_data)
    elif method == "Gaussian Mixture":
        model.fit(scaled_data)
        clusters = model.predict(scaled_data)  # CORRECTION: utiliser predict() au lieu de labels_
    else:
        model.fit(scaled_data)
        clusters = model.labels_
    
    return clusters, model, scaler

def calculate_cluster_metrics(data, clusters):
    """Calcule les métriques de qualité du clustering"""
    if len(np.unique(clusters)) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    
    try:
        silhouette = silhouette_score(data, clusters)
        calinski = calinski_harabasz_score(data, clusters)
        davies = davies_bouldin_score(data, clusters)
        
        return {
            "silhouette": silhouette,
            "calinski_harabasz": calinski,
            "davies_bouldin": davies
        }
    except:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}

def calculate_cluster_matching(clusters, reference_column):
    """Calcule la correspondance entre clusters et colonne de référence"""
    
    # Gérer les types de données
    if reference_column.dtype == 'object' or reference_column.nunique() < 20:
        # Variable catégorielle
        le = LabelEncoder()
        reference_encoded = le.fit_transform(reference_column.fillna('Missing'))
        
        # Calculer l'intersection sur union pour chaque paire cluster/catégorie
        unique_clusters = np.unique(clusters)
        unique_categories = np.unique(reference_encoded)
        
        iou_matrix = np.zeros((len(unique_clusters), len(unique_categories)))
        
        for i, cluster_val in enumerate(unique_clusters):
            cluster_mask = (clusters == cluster_val)
            for j, category_val in enumerate(unique_categories):
                category_mask = (reference_encoded == category_val)
                
                intersection = np.sum(cluster_mask & category_mask)
                union = np.sum(cluster_mask | category_mask)
                
                iou_matrix[i, j] = intersection / union if union > 0 else 0
        
        # Métriques globales
        ari = adjusted_rand_score(reference_encoded, clusters)
        nmi = normalized_mutual_info_score(reference_encoded, clusters)
        
        return {
            "iou_matrix": iou_matrix,
            "unique_clusters": unique_clusters,
            "unique_categories": le.classes_,
            "adjusted_rand_score": ari,
            "normalized_mutual_info": nmi,
            "matching_type": "categorical"
        }
    else:
        # Variable quantitative - utiliser la corrélation entre les moyennes
        cluster_means = []
        category_means = []
        
        unique_clusters = np.unique(clusters)
        
        for cluster_val in unique_clusters:
            cluster_mask = (clusters == cluster_val)
            cluster_means.append(reference_column[cluster_mask].mean())
        
        # Pour les variables quantitatives, on utilise la corrélation de Pearson
        cluster_correlation = np.corrcoef(clusters, reference_column)[0, 1] if len(np.unique(clusters)) > 1 else 0
        
        return {
            "cluster_means": cluster_means,
            "overall_correlation": cluster_correlation,
            "matching_type": "numerical"
        }

def create_3d_cluster_plot(data, x_col, y_col, z_col, clusters, cluster_centers=None):
    """Crée une visualisation 3D des clusters"""
    fig = go.Figure()
    
    unique_clusters = np.unique(clusters)
    colors = px.colors.qualitative.Plotly
    
    for i, cluster_val in enumerate(unique_clusters):
        cluster_data = data[clusters == cluster_val]
        
        fig.add_trace(go.Scatter3d(
            x=cluster_data[x_col],
            y=cluster_data[y_col],
            z=cluster_data[z_col],
            mode='markers',
            marker=dict(
                size=4,
                color=colors[i % len(colors)],
                opacity=0.8
            ),
            name=f'Cluster {cluster_val}',
            text=[f'Cluster {cluster_val}<br>{x_col}: {x:.2f}<br>{y_col}: {y:.2f}<br>{z_col}: {z:.2f}' 
                  for x, y, z in zip(cluster_data[x_col], cluster_data[y_col], cluster_data[z_col])],
            hoverinfo='text'
        ))
    
    if cluster_centers is not None:
        for i, center in enumerate(cluster_centers):
            fig.add_trace(go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)],
                    symbol='diamond'
                ),
                name=f'Centre Cluster {unique_clusters[i]}'
            ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        title="Visualisation 3D des Clusters",
        width=800,
        height=600
    )
    
    return fig

def create_cluster_comparison_heatmap(iou_matrix, cluster_labels, category_labels):
    """Crée une heatmap de comparaison clusters/catégories"""
    fig = px.imshow(
        iou_matrix,
        x=category_labels,
        y=[f"Cluster {c}" for c in cluster_labels],
        color_continuous_scale="Blues",
        title="Correspondance Clusters vs Catégories (IoU)",
        aspect="auto"
    )
    
    fig.update_layout(
        xaxis_title="Catégories de référence",
        yaxis_title="Clusters",
        width=600,
        height=400
    )
    
    # Ajouter les valeurs dans les cellules
    fig.update_traces(text=np.round(iou_matrix, 3), texttemplate="%{text}")
    
    return fig

# =============================================================================
# NOUVELLES FONCTIONS POUR L'ENSEMBLE CLUSTERING ET OPTIMISATION
# =============================================================================

def ensemble_clustering(data, methods_config, n_clusters=3, weights=None):
    """Applique plusieurs méthodes de clustering et combine les résultats"""
    if weights is None:
        weights = [1.0] * len(methods_config)
    
    all_clusterings = []
    method_names = []
    
    for method_name, config in methods_config.items():
        try:
            # S'assurer que n_clusters est inclus dans la configuration
            config_with_n_clusters = config.copy()
            if 'n_clusters' not in config_with_n_clusters:
                config_with_n_clusters['n_clusters'] = n_clusters
                
            clusters, model, scaler = apply_clustering_method(
                data, method_name, **config_with_n_clusters
            )
            
            # Vérifier que nous avons au moins 2 clusters
            unique_clusters = np.unique(clusters)
            if len(unique_clusters) < 2:
                st.warning(f"La méthode {method_name} n'a produit qu'un seul cluster. Elle sera ignorée.")
                continue
                
            all_clusterings.append(clusters)
            method_names.append(method_name)
        except Exception as e:
            st.warning(f"Erreur avec {method_name}: {e}")
            continue
    
    if not all_clusterings:
        raise ValueError("Aucune méthode de clustering n'a fonctionné correctement")
    
    # Matrice de co-occurrence (combien de fois deux points sont dans le même cluster)
    n_samples = len(data)
    cooccurrence_matrix = np.zeros((n_samples, n_samples))
    
    for clusters in all_clusterings:
        for i in range(n_samples):
            for j in range(i, n_samples):
                if clusters[i] == clusters[j]:
                    cooccurrence_matrix[i, j] += 1
                    cooccurrence_matrix[j, i] += 1
    
    # Normaliser par le nombre de méthodes
    cooccurrence_matrix /= len(all_clusterings)
    
    # Clustering hiérarchique sur la matrice de co-occurrence
    # Convertir en dissimilarité
    dissimilarity = 1 - cooccurrence_matrix
    np.fill_diagonal(dissimilarity, 0)
    
    # Vérifier que la dissimilarité n'est pas constante
    if np.all(dissimilarity == 0):
        st.warning("Tous les points sont identiques selon les méthodes de clustering. Utilisation de clusters aléatoires.")
        # Retourner des clusters aléatoires comme solution de secours
        random_clusters = np.random.randint(0, n_clusters, n_samples)
        return random_clusters, all_clusterings, method_names, cooccurrence_matrix
    
    # Clustering hiérarchique
    try:
        condensed_dist = squareform(dissimilarity, checks=False)
        Z = linkage(condensed_dist, method='average')
        
        # Déterminer le nombre optimal de clusters
        max_clusters = min(10, n_samples // 2)
        best_score = -np.inf
        best_clusters = None
        
        for k in range(2, max_clusters + 1):
            clusters = fcluster(Z, k, criterion='maxclust')
            
            # Vérifier que nous avons au moins 2 clusters
            if len(np.unique(clusters)) < 2:
                continue
                
            try:
                score = silhouette_score(dissimilarity, clusters, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_clusters = clusters
            except:
                continue
        
        # Si aucun clustering valide n'a été trouvé, utiliser le nombre de clusters demandé
        if best_clusters is None:
            st.warning("Impossible de trouver un clustering optimal. Utilisation du nombre de clusters spécifié.")
            best_clusters = fcluster(Z, n_clusters, criterion='maxclust')
            
    except Exception as e:
        st.warning(f"Erreur lors du clustering hiérarchique: {e}. Utilisation de KMeans comme solution de secours.")
        # Solution de secours: utiliser KMeans sur la matrice de co-occurrence
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        best_clusters = kmeans.fit_predict(cooccurrence_matrix)
    
    return best_clusters, all_clusterings, method_names, cooccurrence_matrix

def calculate_cluster_agreement_matrix(clusterings):
    """Calcule la matrice d'accord entre différentes méthodes de clustering"""
    n_methods = len(clusterings)
    
    if n_methods == 0:
        return np.array([])
    
    n_samples = len(clusterings[0])
    agreement_matrix = np.zeros((n_methods, n_methods))
    
    for i in range(n_methods):
        for j in range(i, n_methods):
            # Calculer l'ARI entre les deux clusterings
            try:
                ari = adjusted_rand_score(clusterings[i], clusterings[j])
                agreement_matrix[i, j] = ari
                agreement_matrix[j, i] = ari
            except:
                agreement_matrix[i, j] = 0
                agreement_matrix[j, i] = 0
    
    return agreement_matrix

def advanced_multivariate_analysis(df, numeric_cols, color_col=None, clustering_method=None, n_clusters=3, pca_method="récursive"):
    """Analyse multivariée avancée avec PCA et clustering"""
    # Préparer les données
    data = df[numeric_cols].dropna()
    
    if len(data) < 3:
        return None, None, None, None
    
    # Standardisation
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    if pca_method == "récursive":
        # PCA récursif en chaîne avec le nombre correct de composantes finale
        # Beaucoup plus efficace qu'un simple PCA direct 
        component_list = [i for i in range(3, len(numeric_cols))][::-1]
        for n_components in component_list:                            
            pca = PCA(n_components=n_components)
            scaled_data = pca.fit_transform(scaled_data)
        pca_result = scaled_data
        n_components_final = 3
    else:
        # Méthode directe en une seule étape
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(scaled_data)
        n_components_final = 3

    # Créer un DataFrame avec les composantes principales
    pca_columns = [f'PC{i+1}' for i in range(n_components_final)]
    pca_df = pd.DataFrame(data=pca_result, columns=pca_columns)
    
    # Ajouter la colonne de couleur si spécifiée
    if color_col and color_col in df.columns:
        pca_df[color_col] = df.loc[data.index, color_col].values
    
    # Appliquer le clustering si demandé
    clusters = None
    cluster_metrics = None
    
    if clustering_method and clustering_method != "Aucun":
        try:
            clusters, model, _ = apply_clustering_method(
                scaled_data, clustering_method, n_clusters=n_clusters
            )
            pca_df['Cluster'] = clusters
            
            # Calculer les métriques de clustering
            cluster_metrics = calculate_cluster_metrics(scaled_data, clusters)
        except Exception as e:
            st.warning(f"Erreur lors du clustering: {e}")
    
    return pca_df, pca.explained_variance_ratio_, clusters, cluster_metrics

def calculate_cluster_correspondence(clusters1, clusters2, metric='iou'):
    """Calcule la correspondance entre deux ensembles de clusters"""
    if metric == 'iou':
        # Intersection over Union
        unique1 = np.unique(clusters1)
        unique2 = np.unique(clusters2)
        
        iou_matrix = np.zeros((len(unique1), len(unique2)))
        
        for i, c1 in enumerate(unique1):
            mask1 = (clusters1 == c1)
            for j, c2 in enumerate(unique2):
                mask2 = (clusters2 == c2)
                intersection = np.sum(mask1 & mask2)
                union = np.sum(mask1 | mask2)
                iou_matrix[i, j] = intersection / union if union > 0 else 0
        
        return np.max(iou_matrix, axis=1).mean()
    
    elif metric == 'ari':
        # Adjusted Rand Index
        return adjusted_rand_score(clusters1, clusters2)
    
    elif metric == 'nmi':
        # Normalized Mutual Information
        return normalized_mutual_info_score(clusters1, clusters2)
    
    else:
        raise ValueError(f"Métrique non supportée: {metric}")

# =============================================================================
# FONCTIONS POUR L'OPTIMISATION D'HYPERPARAMÈTRES
# =============================================================================

def generate_hyperparameter_combinations(hyperparams_config):
    """Génère toutes les combinaisons possibles d'hyperparamètres"""
    from itertools import product
    
    param_ranges = {}
    for param_name, config in hyperparams_config.items():
        min_val, max_val, step = config['min'], config['max'], config['step']
        # Générer la plage de valeurs avec le pas spécifié
        num_steps = int((max_val - min_val) / step) + 1
        param_ranges[param_name] = np.linspace(min_val, max_val, num_steps)
    
    # Générer toutes les combinaisons
    combinations = list(product(*param_ranges.values()))
    
    # Convertir en liste de dictionnaires
    result = []
    for combo in combinations:
        param_dict = {}
        for i, param_name in enumerate(hyperparams_config.keys()):
            param_dict[param_name] = float(combo[i])  # S'assurer que c'est un float
        result.append(param_dict)
    
    return result

def safe_eval_condition(df, condition, column_mapping):
    """Évalue une condition de manière sécurisée avec mapping des colonnes"""
    try:
        # Remplacer les noms de colonnes génériques par les vrais noms
        condition_with_columns = condition
        for generic_col, real_col in column_mapping.items():
            condition_with_columns = condition_with_columns.replace(generic_col, real_col)
        
        # Remplacer les opérateurs logiques pour la compatibilité avec pandas
        condition_with_columns = condition_with_columns.replace(' and ', ' & ').replace(' or ', ' | ')
        
        # Évaluer la condition
        mask = df.eval(condition_with_columns)
        return mask
    except Exception as e:
        st.warning(f"Erreur dans l'évaluation de la condition: {condition} - {e}")
        return pd.Series([False] * len(df))

def evaluate_metric_on_filtered_data(df, conditions, column_mapping, x_col, y_col, cluster_col=None, metric_expression="pearson + brice"):
    """Évalue une métrique sur les données filtrées selon les conditions données"""
    default_result = {
        'metric_value': -np.inf, 
        'filtered_count': 0, 
        'pearson': -np.inf,
        'brice': -np.inf,
        'brice1': -np.inf,
        'cluster_metrics': {}
    }
    
    try:
        # Appliquer les conditions de filtrage
        filtered_df = df.copy()
        valid_conditions = []
        
        for condition in conditions:
            mask = safe_eval_condition(filtered_df, condition, column_mapping)
            if len(mask) > 0 and mask.any():
                valid_conditions.append(mask)
        
        if valid_conditions:
            combined_mask = np.all(valid_conditions, axis=0)
            filtered_df = filtered_df[combined_mask]
        
        # Vérifier qu'il reste assez de données
        if len(filtered_df) < 2:
            return default_result
        
        # Vérifier que les colonnes X et Y existent et ont des données
        if x_col not in filtered_df.columns or y_col not in filtered_df.columns:
            return default_result
        
        if filtered_df[x_col].isnull().all() or filtered_df[y_col].isnull().all():
            return default_result
        
        # Calculer les corrélations avec gestion des erreurs
        try:
            pearson_corr = filtered_df[x_col].corr(filtered_df[y_col])
            if np.isnan(pearson_corr):
                pearson_corr = -np.inf
        except:
            pearson_corr = -np.inf
        
        try:
            brice_corrs = calculate_brice_correlation(filtered_df[x_col], filtered_df[y_col])
            brice_val = brice_corrs['brice'] if not np.isnan(brice_corrs['brice']) else -np.inf
            brice1_val = brice_corrs['brice1'] if not np.isnan(brice_corrs['brice1']) else -np.inf
        except:
            brice_val = -np.inf
            brice1_val = -np.inf
        
        # Évaluer la métrique personnalisée
        metric_context = {
            'pearson': pearson_corr,
            'brice': brice_val,
            'brice1': brice1_val,
            'count': len(filtered_df)
        }
        
        try:
            metric_value = eval(metric_expression, {"__builtins__": {}}, metric_context)
            if np.isnan(metric_value):
                metric_value = -np.inf
        except:
            metric_value = pearson_corr + brice_val if pearson_corr != -np.inf and brice_val != -np.inf else -np.inf
        
        # Métriques par cluster si une colonne de clustering est spécifiée
        cluster_metrics = {}
        if cluster_col and cluster_col in filtered_df.columns:
            for cluster_val in filtered_df[cluster_col].unique():
                cluster_data = filtered_df[filtered_df[cluster_col] == cluster_val]
                if len(cluster_data) >= 2:
                    try:
                        cluster_pearson = cluster_data[x_col].corr(cluster_data[y_col])
                        if np.isnan(cluster_pearson):
                            cluster_pearson = -np.inf
                    except:
                        cluster_pearson = -np.inf
                    
                    try:
                        cluster_brice = calculate_brice_correlation(cluster_data[x_col], cluster_data[y_col])
                        cluster_brice_val = cluster_brice['brice'] if not np.isnan(cluster_brice['brice']) else -np.inf
                    except:
                        cluster_brice_val = -np.inf
                    
                    cluster_metric_context = {
                        'pearson': cluster_pearson,
                        'brice': cluster_brice_val,
                        'count': len(cluster_data)
                    }
                    
                    try:
                        cluster_metric_value = eval(metric_expression, {"__builtins__": {}}, cluster_metric_context)
                        if np.isnan(cluster_metric_value):
                            cluster_metric_value = -np.inf
                    except:
                        cluster_metric_value = cluster_pearson + cluster_brice_val if cluster_pearson != -np.inf and cluster_brice_val != -np.inf else -np.inf
                    
                    cluster_metrics[cluster_val] = {
                        'metric_value': cluster_metric_value,
                        'pearson': cluster_pearson,
                        'brice': cluster_brice_val,
                        'count': len(cluster_data)
                    }
        
        return {
            'metric_value': metric_value,
            'filtered_count': len(filtered_df),
            'pearson': pearson_corr,
            'brice': brice_val,
            'brice1': brice1_val,
            'cluster_metrics': cluster_metrics
        }
    
    except Exception as e:
        st.error(f"Erreur lors de l'évaluation de la métrique: {e}")
        return default_result

def optimize_hyperparameters(df, conditions_template, column_mapping, hyperparams_config, 
                           x_col, y_col, cluster_col=None, optimization_direction='maximize',
                           max_combinations=1000, metric_expression="pearson + brice"):
    """Optimise les hyperparamètres pour maximiser/minimiser une métrique"""
    results = []
    
    # Générer toutes les combinaisons d'hyperparamètres
    all_combinations = generate_hyperparameter_combinations(hyperparams_config)
    
    # Limiter le nombre de combinaisons si nécessaire
    if len(all_combinations) > max_combinations:
        st.warning(f"Trop de combinaisons ({len(all_combinations)}). Échantillonnage aléatoire de {max_combinations} combinaisons.")
        np.random.shuffle(all_combinations)
        all_combinations = all_combinations[:max_combinations]
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, params in enumerate(all_combinations):
        progress = (i + 1) / len(all_combinations)
        progress_bar.progress(progress)
        status_text.text(f"Évaluation {i+1}/{len(all_combinations)}")
        
        # Remplacer les hyperparamètres dans les conditions
        conditions = []
        for condition_template in conditions_template:
            condition = condition_template
            for param_name, param_value in params.items():
                condition = re.sub(r'\b' + param_name + r'\b', str(param_value), condition)
            conditions.append(condition)
        
        # Évaluer la métrique
        metric_result = evaluate_metric_on_filtered_data(
            df, conditions, column_mapping, x_col, y_col, cluster_col, metric_expression
        )
        
        result_entry = {
            'parameters': params,
            'metric_value': metric_result.get('metric_value', -np.inf),
            'pearson': metric_result.get('pearson', -np.inf),
            'brice': metric_result.get('brice', -np.inf),
            'brice1': metric_result.get('brice1', -np.inf),
            'filtered_count': metric_result.get('filtered_count', 0),
            'cluster_metrics': metric_result.get('cluster_metrics', {})
        }
        
        results.append(result_entry)
    
    progress_bar.empty()
    status_text.empty()
    
    # Filtrer les résultats invalides
    valid_results = [r for r in results if r['metric_value'] != -np.inf and r['filtered_count'] > 0]
    
    if not valid_results:
        st.error("Aucun résultat valide trouvé. Vérifiez vos conditions et paramètres.")
        return []
    
    # Trier les résultats selon la direction d'optimisation
    if optimization_direction == 'maximize':
        valid_results.sort(key=lambda x: x['metric_value'], reverse=True)
    else:
        valid_results.sort(key=lambda x: x['metric_value'])
    
    return valid_results

def create_optimization_interface(df):
    """Crée l'interface pour l'optimisation d'hyperparamètres"""
    st.header("🔎 Optimisation d'Hyperparamètres Avancée")
    
    # Section de configuration de base
    st.subheader("📊 Configuration de base")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_col = st.selectbox("Colonne X", numeric_cols, key="opt_x")
    with col2:
        y_col = st.selectbox("Colonne Y", numeric_cols, key="opt_y")
    with col3:
        cluster_options = [""] + df.columns.tolist()
        cluster_col = st.selectbox("Colonne de clustering (optionnel)", cluster_options, key="opt_cluster")
        if cluster_col == "":
            cluster_col = None
    
    # Configuration du mapping des colonnes
    st.subheader("🔗 Mapping des colonnes génériques")
    
    st.info("""
    **Instructions :** 
    - Utilisez des noms génériques comme `col1`, `col2`, etc. dans vos conditions
    - Mappez chaque colonne générique à une colonne réelle de votre dataset
    - Vous pouvez utiliser autant de colonnes que nécessaire
    """)
    
    # Interface pour ajouter des mappings de colonnes
    column_mapping = st.session_state.get('column_mapping', {})
    
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        new_generic = st.text_input("Nom générique (ex: col1)", key="new_generic")
    with col2:
        new_real = st.selectbox("Colonne réelle", [""] + df.columns.tolist(), key="new_real")
    with col3:
        if st.button("➕ Ajouter mapping", width='stretch') and new_generic and new_real:
            column_mapping[new_generic] = new_real
            st.session_state.column_mapping = column_mapping
            st.rerun()
    
    # Afficher les mappings existants
    if column_mapping:
        st.write("**Mappings actuels :**")
        for generic, real in list(column_mapping.items()):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.text(f"{generic} → {real}")
            with col2:
                if st.button("✏️ Modifier", key=f"edit_{generic}"):
                    pass
            with col3:
                if st.button("🗑️ Supprimer", key=f"del_{generic}"):
                    del column_mapping[generic]
                    st.session_state.column_mapping = column_mapping
                    st.rerun()
    
    # Configuration des conditions
    st.subheader("📝 Conditions avec Hyperparamètres")
    
    conditions = st.session_state.get('optimization_conditions', [])
    hyperparams_config = st.session_state.get('hyperparams_config', {})
    
    # Interface pour ajouter des conditions
    with st.expander("➕ Ajouter une nouvelle condition", expanded=True):
        st.info("""
        **Syntaxe des conditions :**
        - Utilisez les noms génériques définis ci-dessus (ex: `col1`, `col2`)
        - Utilisez des hyperparamètres comme `a`, `b`, `c` (lettres simples)
        - Exemple : `(abs(col1 - col2) < a) and (col3 * col4 > b) or (col5 <= c)`
        """)
        
        new_condition = st.text_area(
            "Nouvelle condition", 
            value="abs(col1 - col2) < a",
            height=80,
            key="new_condition"
        )
        
        # Détection automatique des hyperparamètres
        if new_condition:
            param_pattern = r'\b([a-zA-Z])\b'
            detected_params = set(re.findall(param_pattern, new_condition))
            
            generic_cols = set(column_mapping.keys())
            hyperparams = [p for p in detected_params if p not in generic_cols and len(p) == 1]
            
            if hyperparams:
                st.write("**Hyperparamètres détectés :**", ", ".join(hyperparams))
                
                for param in hyperparams:
                    st.write(f"**Configuration de '{param}'**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        min_val = st.number_input(f"Valeur minimale", value=0.0, key=f"min_{param}")
                    with col2:
                        max_val = st.number_input(f"Valeur maximale", value=1.0, key=f"max_{param}")
                    with col3:
                        step = st.number_input(f"Pas d'incrémentation", value=0.1, key=f"step_{param}")
                    
                    hyperparams_config[param] = {'min': min_val, 'max': max_val, 'step': step}
        
        if st.button("✅ Ajouter cette condition", width='stretch'):
            if new_condition.strip():
                conditions.append(new_condition.strip())
                st.session_state.optimization_conditions = conditions
                st.session_state.hyperparams_config = hyperparams_config
                st.success("Condition ajoutée!")
                st.rerun()
    
    # Affichage des conditions actives
    if conditions:
        st.subheader("✅ Conditions actives")
        for i, condition in enumerate(conditions):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.code(condition, language='python')
            with col2:
                if st.button("🗑️", key=f"del_cond_{i}"):
                    conditions.pop(i)
                    st.session_state.optimization_conditions = conditions
                    st.rerun()
    
    # Configuration de l'optimisation
    st.subheader("⚙️ Configuration de l'Optimisation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        optimization_direction = st.selectbox("Direction d'optimisation", 
                                           ["maximize", "minimize"])
    with col2:
        max_results = st.number_input("Nombre de meilleurs résultats à afficher", 
                                    min_value=1, max_value=50, value=10)
    with col3:
        max_combinations = st.number_input("Nombre maximum de combinaisons", 
                                         min_value=10, max_value=10000, value=500)
    
    metric_expression = st.text_input("Expression de la métrique à optimiser", 
                                    value="pearson + brice")
    
    # Validation avant lancement
    if not conditions:
        st.error("❌ Veuillez ajouter au moins une condition")
        return
    
    if not column_mapping:
        st.error("❌ Veuillez configurer le mapping des colonnes")
        return
    
    if not hyperparams_config:
        st.error("❌ Aucun hyperparamètre détecté dans les conditions")
        return
    
    if x_col == y_col:
        st.error("❌ Les colonnes X et Y doivent être différentes")
        return
    
    # Bouton de lancement de l'optimisation
    if st.button("🚀 Lancer l'optimisation", type="primary", width='stretch'):
        with st.spinner("Optimisation en cours... Cette opération peut prendre plusieurs minutes"):
            results = optimize_hyperparameters(
                df, conditions, column_mapping, hyperparams_config, 
                x_col, y_col, cluster_col, optimization_direction, 
                max_combinations, metric_expression
            )
        
        # Affichage des résultats
        st.subheader("📊 Résultats de l'Optimisation")
        
        if not results or results[0]['metric_value'] == -np.inf:
            st.error("Aucun résultat valide trouvé. Vérifiez vos conditions et paramètres.")
            return
        
        # Préparer le tableau des résultats
        results_df_data = []
        for i, result in enumerate(results[:max_results]):
            if result['metric_value'] == -np.inf:
                continue
                
            row = {
                'Rang': i + 1,
                'Métrique': f"{result['metric_value']:.4f}",
                'Pearson': f"{result['pearson']:.4f}" if not np.isnan(result['pearson']) else 'N/A',
                'Brice': f"{result['brice']:.4f}" if not np.isnan(result['brice']) else 'N/A',
                'Points conservés': result['filtered_count']
            }
            for param_name, param_value in result['parameters'].items():
                row[param_name] = f"{param_value:.4f}"
            results_df_data.append(row)
        
        results_df = pd.DataFrame(results_df_data)
        st.dataframe(results_df, width='stretch')
        
        # Graphique pour le meilleur résultat
        best_result = results[0]
        
        best_conditions = []
        for condition_template in conditions:
            condition = condition_template
            for param_name, param_value in best_result['parameters'].items():
                condition = re.sub(r'\b' + param_name + r'\b', str(param_value), condition)
            best_conditions.append(condition)
        
        filtered_df = df.copy()
        valid_conditions = []
        
        for condition in best_conditions:
            mask = safe_eval_condition(filtered_df, condition, column_mapping)
            valid_conditions.append(mask)
        
        if valid_conditions:
            combined_mask = np.all(valid_conditions, axis=0)
            filtered_df = filtered_df[combined_mask]
        
        st.subheader("📈 Visualisation du meilleur résultat")
        
        if cluster_col:
            fig = px.scatter(filtered_df, x=x_col, y=y_col, color=cluster_col,
                           title=f"Meilleur résultat - {x_col} vs {y_col}",
                           hover_data=list(best_result['parameters'].keys()))
        else:
            fig = px.scatter(filtered_df, x=x_col, y=y_col,
                           title=f"Meilleur résultat - {x_col} vs {y_col}")
        
        try:
            fig.add_traces(px.scatter(filtered_df, x=x_col, y=y_col, trendline="ols").data[1])
        except:
            pass
        
        st.plotly_chart(fig, width='stretch')
        
        # Métriques par cluster
        if cluster_col and best_result['cluster_metrics']:
            st.subheader("📋 Métriques par Cluster")
            cluster_data = []
            for cluster_val, metrics in best_result['cluster_metrics'].items():
                cluster_data.append({
                    'Cluster': cluster_val,
                    'Métrique': f"{metrics['metric_value']:.4f}",
                    'Pearson': f"{metrics['pearson']:.4f}" if not np.isnan(metrics['pearson']) else 'N/A',
                    'Brice': f"{metrics['brice']:.4f}" if not np.isnan(metrics['brice']) else 'N/A',
                    'Nombre de points': metrics['count']
                })
            cluster_df = pd.DataFrame(cluster_data)
            st.dataframe(cluster_df, width='stretch')
        
        # Conditions appliquées
        st.subheader("🔧 Conditions appliquées")
        for i, condition in enumerate(best_conditions):
            st.write(f"**Condition {i+1}:** `{condition}`")
        
        # Paramètres optimaux
        st.subheader("⚙️ Paramètres optimaux")
        param_df = pd.DataFrame([best_result['parameters']]).T
        param_df.columns = ['Valeur optimale']
        st.dataframe(param_df, width='stretch')
        
        # Téléchargement des résultats
        st.subheader("💾 Export des résultats")
        
        export_data = []
        for result in results[:max_results]:
            if result['metric_value'] == -np.inf:
                continue
                
            row = {
                'Rang': len(export_data) + 1,
                'Métrique': result['metric_value'],
                'Pearson': result['pearson'],
                'Brice': result['brice'],
                'Points_conservés': result['filtered_count']
            }
            for param_name, param_value in result['parameters'].items():
                row[param_name] = param_value
            export_data.append(row)
        
        export_df = pd.DataFrame(export_data)
        csv_data = export_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Télécharger tous les résultats (CSV)",
            data=csv_data,
            file_name="resultats_optimisation_complets.csv",
            mime="text/csv"
        )

# =============================================================================
# NOUVELLES FONCTIONS POUR LA RÉGRESSION AVANCÉE
# =============================================================================

def create_regression_interface(df):
    """Crée l'interface pour la régression avancée"""
    st.header("📈 Régression Avancée")
    
    # Sélection des variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("❌ La régression nécessite au moins 2 colonnes numériques")
        return
    
    st.subheader("🎯 Configuration de la Régression")
    
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Variable cible (Y)", numeric_cols, key="reg_target")
    with col2:
        feature_cols = st.multiselect("Variables explicatives (X)", numeric_cols, 
                                    default=numeric_cols[:min(5, len(numeric_cols))],
                                    key="reg_features")
    
    if not feature_cols:
        st.error("❌ Sélectionnez au moins une variable explicative")
        return
    
    # Type de régression
    st.subheader("🔧 Type de Régression")
    
    regression_type = st.radio("Type de régression", 
                              ["Linéaire", "Polynomiale", "Personnalisée", "AutoML"],
                              horizontal=True)
    
    if regression_type == "Linéaire":
        model_type = st.selectbox("Modèle linéaire", 
                                ["Régression Linéaire", "Ridge", "Lasso", "ElasticNet"])
        
        if model_type in ["Ridge", "Lasso", "ElasticNet"]:
            alpha = st.slider("Paramètre de régularisation (alpha)", 0.0, 10.0, 1.0, 0.1)
    
    elif regression_type == "Polynomiale":
        degree = st.slider("Degré polynomial", 1, 10, 2)
        model_type = st.selectbox("Modèle de base", 
                                ["Régression Linéaire", "Ridge", "Lasso"])
        if model_type in ["Ridge", "Lasso", "ElasticNet"]:
            alpha = st.slider("Paramètre de régularisation (alpha)", 0.0, 10.0, 1.0, 0.1)
    
    elif regression_type == "Personnalisée":
        st.info("""
        **Syntaxe de la fonction personnalisée :**
        - Utilisez `x1`, `x2`, ... pour les variables explicatives
        - Utilisez `a`, `b`, `c`, ... pour les paramètres à optimiser
        - Exemple : `a * x1 + b * x2**2 + c * exp(x3)`
        """)
        
        custom_function = st.text_area("Fonction personnalisée", 
                                     "a * x1 + b * x2 + c",
                                     height=100)
        
        # Détection des paramètres
        param_pattern = r'\b([a-zA-Z])\b'
        detected_params = set(re.findall(param_pattern, custom_function))
        
        # Exclure les noms de variables (x1, x2, etc.)
        feature_pattern = r'\bx\d+\b'
        feature_names = set(re.findall(feature_pattern, custom_function))
        params = [p for p in detected_params if p not in [name[0] for name in feature_names]]
        
        if params:
            st.write("**Paramètres détectés :**", ", ".join(params))
            
            # Configuration des bornes pour chaque paramètre
            param_bounds = {}
            for param in params:
                col1, col2 = st.columns(2)
                with col1:
                    min_val = st.number_input(f"Min {param}", value=-10.0, key=f"min_{param}")
                with col2:
                    max_val = st.number_input(f"Max {param}", value=10.0, key=f"max_{param}")
                param_bounds[param] = (min_val, max_val)
    
    elif regression_type == "AutoML":
        st.info("🤖 L'AutoML teste automatiquement plusieurs modèles et sélectionne le meilleur")
        models_to_test = st.multiselect("Modèles à tester",
                                      ["Linear", "Random Forest", "Gradient Boosting", "SVR", "Neural Network"],
                                      default=["Linear", "Random Forest", "Gradient Boosting"])
    
    # Options avancées
    with st.expander("⚙️ Options avancées"):
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Taille de l'ensemble de test", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Seed aléatoire", value=42)
        with col2:
            cv_folds = st.slider("Nombre de folds pour validation croisée", 2, 10, 5)
            scoring = st.selectbox("Métrique d'évaluation", 
                                ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"])
    
    # Bouton d'exécution
    if st.button("🚀 Exécuter la régression", type="primary"):
        if len(feature_cols) == 0:
            st.error("❌ Sélectionnez au moins une variable explicative")
            return
        
        # Préparer les données
        data = df[feature_cols + [target_col]].dropna()
        
        if len(data) < 10:
            st.error("❌ Pas assez de données valides pour la régression")
            return
        
        X = data[feature_cols]
        y = data[target_col]
        
        # Séparation train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        with st.spinner("Entraînement du modèle en cours..."):
            if regression_type == "Linéaire":
                if model_type == "Régression Linéaire":
                    model = LinearRegression()
                elif model_type == "Ridge":
                    model = Ridge(alpha=alpha)
                elif model_type == "Lasso":
                    model = Lasso(alpha=alpha)
                elif model_type == "ElasticNet":
                    model = ElasticNet(alpha=alpha)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif regression_type == "Polynomiale":
                # Créer les features polynomiales
                poly = PolynomialFeatures(degree=degree)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                
                if model_type == "Régression Linéaire":
                    model = LinearRegression()
                elif model_type == "Ridge":
                    model = Ridge(alpha=alpha)
                elif model_type == "Lasso":
                    model = Lasso(alpha=alpha)
                
                model.fit(X_train_poly, y_train)
                y_pred = model.predict(X_test_poly)
                
            elif regression_type == "Personnalisée":
                # Optimisation non linéaire pour la fonction personnalisée
                def objective(params):
                    # Créer un dictionnaire paramètre:valeur
                    param_dict = {p: v for p, v in zip(params, params)}
                    
                    # Évaluer la fonction pour chaque ligne
                    predictions = []
                    for i in range(len(X)):
                        # Créer le contexte d'évaluation
                        context = param_dict.copy()
                        for j, col in enumerate(feature_cols):
                            context[f'x{j+1}'] = X.iloc[i][col]
                        
                        try:
                            pred = eval(custom_function, {"__builtins__": {}}, context)
                            predictions.append(pred)
                        except:
                            return np.inf
                    
                    # Calculer l'erreur quadratique moyenne
                    mse = np.mean((np.array(predictions) - y.values) ** 2)
                    return mse
                
                # Bornes initiales pour les paramètres
                initial_guess = [0.0] * len(params)
                bounds = [param_bounds[p] for p in params]
                
                # Optimisation
                result = opt.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
                
                if result.success:
                    best_params = {p: v for p, v in zip(params, result.x)}
                    st.success("✅ Optimisation réussie!")
                    
                    # Calculer les prédictions avec les paramètres optimaux
                    y_pred = []
                    for i in range(len(X_test)):
                        context = best_params.copy()
                        for j, col in enumerate(feature_cols):
                            context[f'x{j+1}'] = X_test.iloc[i][col]
                        y_pred.append(eval(custom_function, {"__builtins__": {}}, context))
                    y_pred = np.array(y_pred)
                    
                    model = type('CustomModel', (), {'params': best_params, 'function': custom_function})()
                else:
                    st.error("❌ Échec de l'optimisation")
                    return
                
            elif regression_type == "AutoML":
                # Tester plusieurs modèles
                models = {
                    "Linear": LinearRegression(),
                    "Random Forest": RandomForestRegressor(random_state=random_state),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
                    "SVR": SVR(),
                    "Neural Network": MLPRegressor(random_state=random_state, max_iter=1000)
                }
                
                best_score = -np.inf
                best_model = None
                best_name = ""
                
                results = []
                for name in models_to_test:
                    if name in models:
                        model = models[name]
                        
                        # Validation croisée
                        cv_scores = cross_val_score(model, X_train, y_train, 
                                                  cv=cv_folds, scoring=scoring)
                        mean_score = np.mean(cv_scores)
                        
                        results.append({
                            'Modèle': name,
                            'Score CV': f"{mean_score:.4f}",
                            'Écart-type': f"{np.std(cv_scores):.4f}"
                        })
                        
                        if mean_score > best_score:
                            best_score = mean_score
                            best_model = model
                            best_name = name
                
                # Entraîner le meilleur modèle sur tout l'ensemble d'entraînement
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                model = best_model
                
                st.success(f"✅ Meilleur modèle : {best_name} (score: {best_score:.4f})")
                
                # Afficher les résultats de tous les modèles
                st.subheader("📊 Comparaison des modèles")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, width='stretch')
        
        # Évaluation du modèle
        st.subheader("📈 Performance du Modèle")
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R²", f"{r2:.4f}")
        with col2:
            st.metric("MSE", f"{mse:.4f}")
        with col3:
            st.metric("MAE", f"{mae:.4f}")
        
        # Visualisation des résultats
        st.subheader("👁️ Visualisation des Prédictions")
        
        results_df = pd.DataFrame({
            'Valeur Réelle': y_test,
            'Prédiction': y_pred,
            'Résidu': y_test - y_pred
        })
        
        # Graphique des prédictions vs valeurs réelles
        fig1 = px.scatter(results_df, x='Valeur Réelle', y='Prédiction',
                         title="Prédictions vs Valeurs Réelles",
                         trendline="ols")
        fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                y=[y_test.min(), y_test.max()],
                                mode='lines', name='Ligne parfaite',
                                line=dict(color='red', dash='dash')))
        st.plotly_chart(fig1, width='stretch')
        
        # Graphique des résidus
        fig2 = px.scatter(results_df, x='Prédiction', y='Résidu',
                         title="Analyse des Résidus")
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2, width='stretch')
        
        # Distribution des résidus
        fig3 = px.histogram(results_df, x='Résidu', 
                           title="Distribution des Résidus",
                           nbins=30)
        st.plotly_chart(fig3, width='stretch')
        
        # Coefficients du modèle (pour les modèles linéaires)
        if hasattr(model, 'coef_'):
            st.subheader("📋 Coefficients du Modèle")
            
            if regression_type == "Polynomiale":
                # Pour les modèles polynomiales, obtenir les noms des features
                feature_names = poly.get_feature_names_out(feature_cols)
                coefficients = model.coef_
            else:
                feature_names = feature_cols
                coefficients = model.coef_
            
            if hasattr(model, 'intercept_'):
                intercept = model.intercept_
                coeff_df = pd.DataFrame({
                    'Feature': ['Intercept'] + list(feature_names),
                    'Coefficient': [intercept] + list(coefficients)
                })
            else:
                coeff_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients
                })
            
            st.dataframe(coeff_df, width='stretch')
            
            # Graphique des coefficients
            fig4 = px.bar(coeff_df, x='Feature', y='Coefficient',
                         title="Importance des Variables (Coefficients)")
            st.plotly_chart(fig4, width='stretch')
        
        elif regression_type == "Personnalisée":
            st.subheader("📋 Paramètres Optimaux")
            param_df = pd.DataFrame(list(model.params.items()), 
                                  columns=['Paramètre', 'Valeur'])
            st.dataframe(param_df, width='stretch')
            
            st.write("**Fonction optimisée :**")
            st.code(custom_function)
        
        # Importance des features (pour les modèles d'ensemble)
        elif hasattr(model, 'feature_importances_'):
            st.subheader("📊 Importance des Variables")
            
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(importance_df, width='stretch')
            
            fig5 = px.bar(importance_df, x='Feature', y='Importance',
                         title="Importance des Variables")
            st.plotly_chart(fig5, width='stretch')
        
        # Téléchargement des résultats
        st.subheader("💾 Export des Résultats")
        
        # Ajouter les prédictions au DataFrame original
        results_full = df.copy()
        results_full['Prediction'] = model.predict(X) if hasattr(model, 'predict') else np.nan
        results_full['Residu'] = results_full[target_col] - results_full['Prediction']
        
        csv_data = results_full.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les prédictions (CSV)",
            data=csv_data,
            file_name="predictions_regression.csv",
            mime="text/csv"
        )

# =============================================================================
# FONCTIONS POUR LES INFOBULLES ET DICTIONNAIRE DE MÉTRIQUES
# =============================================================================

def create_tooltip(help_text):
    """Crée une infobulle avec le texte d'aide"""
    st.markdown(f'<span title="{help_text}" style="border-bottom: 1px dotted #0077cc; cursor: help;">ℹ️</span>', unsafe_allow_html=True)

def show_metric_info(metric_name):
    """Affiche les informations détaillées sur une métrique spécifique"""
    
    # Utiliser r"""...""" pour les formules LaTeX sur plusieurs lignes
    metrics_info = {
        "Pearson": {
            "description": "Mesure la corrélation linéaire entre deux variables continues.",
            "formula": r"""
                \begin{align*}
                r &= \frac{\sum_{i=1}^{n}[(x_i - \bar{x})(y_i - \bar{y})]}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}} \\
                \text{où } \bar{x} &\text{ est la moyenne de } x, \bar{y} \text{ la moyenne de } y.
                \end{align*}
            """,
            "interpretation": "Valeurs entre -1 et 1. 1: corrélation positive parfaite, -1: corrélation négative parfaite, 0: absence de corrélation linéaire.",
            "usage": "Idéal pour détecter les relations linéaires entre variables normalement distribuées."
        },
        "Brice": {
            "description": "Corrélation basée sur les variations premières (dérivées discrètes) des séries temporelles.",
            "formula": r"""
                \begin{align*}
                \text{Brice} &= \frac{\sum_{i=2}^{n}(\Delta x_i \cdot \Delta y_i)}{\sqrt{\sum_{i=2}^{n}(\Delta x_i)^2 \cdot \sum_{i=2}^{n}(\Delta y_i)^2}} \\
                \text{où } \Delta x_i &= x_i - x_{i-1}
                \end{align*}
            """,
            "interpretation": "Mesure la similarité dans les tendances à court terme. Valeurs élevées indiquent que les variables évoluent dans la même direction.",
            "usage": "Utile pour l'analyse de séries temporelles et la détection de co-mouvements."
        },
        "Brice1": {
            "description": "Version centrée de la corrélation Brice, utilisant les variations centrées.",
            "formula": r"""
                \begin{align*}
                \text{Brice1} &= \frac{\sum_{i=2}^{n}[(\Delta x_i - \mu_{\Delta x})(\Delta y_i - \mu_{\Delta y})]}{\sqrt{\sum_{i=2}^{n}(\Delta x_i - \mu_{\Delta x})^2 \cdot \sum_{i=2}^{n}(\Delta y_i - \mu_{\Delta y})^2}} \\
                \text{où } \mu_{\Delta x} &\text{ est la moyenne des variations de } x, \mu_{\Delta y} \text{ celle de } y.
                \end{align*}
            """,
            "interpretation": "Similaire à Brice mais moins sensible aux tendances globales, se concentrant sur les variations relatives.",
            "usage": "Recommandé lorsque les séries présentent des tendances non stationnaires."
        },
        "Silhouette": {
            "description": "Mesure la qualité de la séparation entre les clusters.",
            "formula": r"""
                \begin{align*}
                s(i) &= \frac{b(i) - a(i)}{\max\{a(i), b(i)\}} \\
                \text{où } a(i) &\text{ est la distance moyenne intra-cluster,} \\
                b(i) &\text{ est la distance moyenne au cluster le plus proche.}
                \end{align*}
            """,
            "interpretation": "Valeurs entre -1 et 1. Valeurs proches de 1: clusters bien séparés, proches de 0: clusters se chevauchent, négatives: points mal classés.",
            "usage": "Critère de validation pour le clustering, indépendant du nombre de clusters."
        },
        "Calinski-Harabasz": {
            "description": "Ratio entre la dispersion inter-cluster et intra-cluster.",
            "formula": r"""
                \begin{align*}
                CH &= \frac{B/(k-1)}{W/(n-k)} \\
                \text{où } B &\text{ est la somme des carrés inter-clusters,} \\
                W &\text{ est la somme des carrés intra-clusters,} \\
                n &\text{ le nombre de points, } k \text{ le nombre de clusters.}
                \end{align*}
            """,
            "interpretation": "Valeurs plus élevées indiquent une meilleure séparation entre clusters. Pas de borne supérieure fixe.",
            "usage": "Utile pour comparer différentes configurations de clustering sur le même dataset."
        },
        "Davies-Bouldin": {
            "description": "Mesure la similarité moyenne entre chaque cluster et son cluster le plus similaire.",
            "formula": r"""
                \begin{align*}
                DB &= \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left\{ \frac{s_i + s_j}{d(c_i, c_j)} \right\} \\
                \text{où } s_i &\text{ est la dispersion moyenne du cluster } i.
                \end{align*}
            """,
            "interpretation": "Valeurs plus faibles indiquent une meilleure séparation. Minimum théorique de 0.",
            "usage": "Bon pour les clusters de formes et densités variées."
        },
        "ARI": {
            "description": "Adjusted Rand Index - Mesure la similarité entre deux partitions, corrigée du hasard.",
            "formula": r"""
                \begin{align*}
                ARI &= \frac{\text{Index} - \text{Expected\_Index}}{\text{Max\_Index} - \text{Expected\_Index}}
                \end{align*}
            """,
            "interpretation": "Valeurs entre -1 et 1. 1: accord parfait, 0: accord aléatoire, négatif: accord pire que le hasard.",
            "usage": "Validation externe du clustering lorsque la vérité terrain est disponible."
        },
        "NMI": {
            "description": "Normalized Mutual Information - Mesure l'information mutuelle normalisée entre deux partitions.",
            "formula": r"""
                \begin{align*}
                NMI &= \frac{I(X;Y)}{\sqrt{H(X)H(Y)}} \\
                \text{où } I(X;Y) &\text{ est l'Information Mutuelle, } \\
                H(X) \text{ et } H(Y) &\text{ sont les Entropies des partitions.}
                \end{align*}
            """,
            "interpretation": "Valeurs entre 0 et 1. 1: accord parfait, 0: indépendance statistique.",
            "usage": "Comparaison de partitions avec différents nombres de clusters."
        },
        "Entropie (Shannon)": {
            "description": "Mesure l'incertitude ou la quantité d'information contenue dans une partition.",
            "formula": r"""
                \begin{align*}
                H(X) &= -\sum_{i=1}^{|C_X|} p(x_i) \log_2(p(x_i)) \\
                \text{où } p(x_i) &= \frac{|C_i|}{|N|} \\
                |C_X| &\text{ est le nombre de clusters dans la partition } X.
                \end{align*}
            """,
            "interpretation": "Une valeur plus élevée indique une partition plus uniformément distribuée (plus incertaine).",
            "usage": "Bloc de construction fondamental pour toutes les mesures basées sur l'information."
        },
        "Information Mutuelle": {
            "description": "Mesure la quantité d'information partagée entre deux partitions (dépendance mutuelle).",
            "formula": r"""
                \begin{align*}
                I(X;Y) &= \sum_{i=1}^{|C_X|} \sum_{j=1}^{|C_Y|} p(x_i, y_j) \log_2 \left( \frac{p(x_i, y_j)}{p(x_i) p(y_j)} \right) \\
                \text{où } p(x_i, y_j) &\text{ est la probabilité conjointe, } \\
                p(x_i) &\text{ et } p(y_j) \text{ sont les probabilités marginales.}
                \end{align*}
            """,
            "interpretation": "Plus la valeur est élevée, plus l'accord entre les deux partitions est fort (0 = indépendance totale).",
            "usage": "Utilisé comme numérateur pour le NMI (Normalized Mutual Information)."
        },
        "Variation de l'Information": {
            "description": "Mesure la distance entre deux partitions. C'est une alternative au NMI.",
            "formula": r"""
                \begin{align*}
                VI(X, Y) &= H(X|Y) + H(Y|X) \\
                &= H(X) + H(Y) - 2 I(X;Y) \\
                \text{où } H(X|Y) &\text{ est l'entropie conditionnelle de } X \text{ sachant } Y.
                \end{align*}
            """,
            "interpretation": "Une valeur de 0 indique un accord parfait (partitions identiques). Plus la valeur est faible, meilleur est le clustering.",
            "usage": "Validation externe du clustering lorsque la vérité terrain est disponible."
        },
        "R² (R-carré)": {
            "description": "Coefficient de détermination - Mesure la proportion de variance expliquée par le modèle.",
            "formula": r"""
                \begin{align*}
                R^2 &= 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \\
                \text{où } \hat{y}_i &\text{ est la prédiction, } \bar{y} \text{ la moyenne des observations.}
                \end{align*}
            """,
            "interpretation": "Valeurs entre 0 et 1. 1: modèle parfait, 0: modèle n'explique aucune variance.",
            "usage": "Métrique principale pour évaluer les modèles de régression."
        },
        "MSE": {
            "description": "Mean Squared Error - Erreur quadratique moyenne.",
            "formula": r"""
                \begin{align*}
                MSE &= \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
                \end{align*}
            """,
            "interpretation": "Plus la valeur est faible, meilleur est le modèle. Sensible aux valeurs extrêmes.",
            "usage": "Métrique courante pour la régression, pénalise fortement les grandes erreurs."
        },
        "MAE": {
            "description": "Mean Absolute Error - Erreur absolue moyenne.",
            "formula": r"""
                \begin{align*}
                MAE &= \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
                \end{align*}
            """,
            "interpretation": "Plus la valeur est faible, meilleur est le modèle. Moins sensible aux outliers que le MSE.",
            "usage": "Bonne alternative au MSE quand les données contiennent des valeurs aberrantes."
        },
        "RMSE": {
            "description": "Root Mean Squared Error - Racine carrée de l'erreur quadratique moyenne.",
            "formula": r"""
                \begin{align*}
                RMSE &= \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
                \end{align*}
            """,
            "interpretation": "Dans la même unité que la variable cible. Plus la valeur est faible, meilleur est le modèle.",
            "usage": "Facilite l'interprétation en conservant l'unité de mesure originale."
        }
    }
    
    if metric_name in metrics_info:
        info = metrics_info[metric_name]
        st.markdown(f"### {metric_name}")
        st.markdown(f"**Description:** {info['description']}")
        
        # Affichage avec st.latex()
        st.markdown("**Formule:**")
        st.latex(info['formula']) 
        
        st.markdown(f"**Interprétation:** {info['interpretation']}")
        st.markdown(f"**Usage recommandé:** {info['usage']}")
    else:
        st.warning(f"Métrique '{metric_name}' non trouvée dans le dictionnaire.")

def create_metrics_dictionary():
    """Crée le dictionnaire complet des métriques disponibles"""
    st.header("📚 Dictionnaire des Métriques")
    
    # Métriques de corrélation
    with st.expander("📈 Métriques de Corrélation", width="stretch", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Pearson", width="stretch"):
                show_metric_info("Pearson")
        with col2:
            if st.button("Brice", width="stretch"):
                show_metric_info("Brice")
        with col3:
            if st.button("Brice1", width="stretch"):
                show_metric_info("Brice1")
    
    # Métriques de clustering
    with st.expander("🔍 Métriques de Clustering", width="stretch", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Silhouette", width="stretch"):
                show_metric_info("Silhouette")
        with col2:
            if st.button("Calinski-Harabasz", width="stretch"):
                show_metric_info("Calinski-Harabasz")
        with col3:
            if st.button("Davies-Bouldin", width="stretch"):
                show_metric_info("Davies-Bouldin")
        with col4:
            if st.button("ARI", width="stretch"):
                show_metric_info("ARI")
        
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            if st.button("NMI", width="stretch"):
                show_metric_info("NMI")
        with col6:
            if st.button("Entropie (Shannon)", width="stretch"):
                show_metric_info("Entropie (Shannon)")
        with col7:
            if st.button("Information Mutuelle", width="stretch"):
                show_metric_info("Information Mutuelle")
        with col8:
            if st.button("Variation de l'Information", width="stretch"):
                show_metric_info("Variation de l'Information")
        
    
    # Métriques de régression
    with st.expander("📊 Métriques de Régression", width="stretch", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("R² (R-carré)", width="stretch"):
                show_metric_info("R² (R-carré)")
        with col2:
            if st.button("MSE", width="stretch"):
                show_metric_info("MSE")
        with col3:
            if st.button("MAE", width="stretch"):
                show_metric_info("MAE")
        with col4:
            if st.button("RMSE", width="stretch"):
                show_metric_info("RMSE")
    
    # Recherche de métrique
    st.subheader("🔍 Recherche de Métrique")
    search_term = st.text_input("Entrez le nom d'une métrique")
    if search_term:
        show_metric_info(search_term)

# =============================================================================
# INTERFACE STREAMLIT AMÉLIORÉE
# =============================================================================

def initialize_session_state():
    """Initialise l'état de session avec toutes les variables nécessaires"""
    defaults = {
        'uploaded_files': {},
        'original_data': {},
        'filtered_data': {},
        'filters': {},
        'advanced_filters': {},
        'id_column': None,
        'sampling_file': None,
        'theme_settings': {
            'font_family': 'Arial', 'font_size': 14, 'color_theme': 'plotly_white',
            'app_theme': 'light', 'primary_color': '#1f77b4',
            'background_color': '#ffffff', 'text_color': '#000000'
        },
        'custom_css': "", 
        'max_categories': 20, 
        'ignored_jobs': {'count': 0, 'total_jobs': 0, 'reason': {}},
        'max_graphs_per_row': 2, 
        'analysis_results': {}, 
        'current_tab': 'Distribution',
        'graph_settings': {'width': 800, 'height': 600, 'title_size': 16, 'axis_size': 14, 'legend_size': 12},
        'all_columns': [],
        'missing_threshold': 60,
        'filter_mode': 'predefined',
        'heatmap_columns': None,
        'data_loaded': False,
        'column_mapping': {},
        'optimization_conditions': [],
        'hyperparams_config': {},
        'clustering_results': None,
        'cluster_comparison_results': None,
        'separator': ',',
        'logo_path': "Images/Logo/Logo_BKZ_1_5.png",
        'data_source': 'Fichiers locaux',  # CORRECTION: Utiliser la même valeur que dans le radio
        'sklearn_dataset': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def apply_custom_css(css):
    """Applique du CSS personnalisé à l'application"""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def display_logo():
    """Affiche le logo de l'application"""
    if st.session_state.logo_path and os.path.exists(st.session_state.logo_path):
        try:
            st.sidebar.image(st.session_state.logo_path, width="stretch")
        except:
            st.sidebar.info("❌ Impossible de charger le logo")
    else:
        # Logo par défaut ou espace réservé
        st.sidebar.markdown("### 📊 IA Analytics")

def create_sidebar():
    """Crée la sidebar avec toutes les options de configuration"""
    with st.sidebar:
        # Affichage du logo
        display_logo()
        
        st.markdown("---")
        st.header("⚙️ Configuration Avancée")
        
        with st.expander("🎨 Personnalisation du Thème", expanded=False):
            app_theme = st.selectbox("Thème de l'application", ["Light", "Dark", "Custom"])
            create_tooltip("Choisissez l'apparence générale de l'application")
            
            if app_theme == "Custom":
                col1, col2 = st.columns(2)
                with col1:
                    primary_color = st.color_picker("Couleur principale", "#1f77b4")
                    create_tooltip("Couleur dominante utilisée pour les boutons et titres")
                    background_color = st.color_picker("Arrière-plan", "#ffffff")
                    create_tooltip("Couleur de fond de l'application")
                with col2:
                    text_color = st.color_picker("Couleur du texte", "#000000")
                    create_tooltip("Couleur du texte principal")
                    secondary_color = st.color_picker("Couleur secondaire", "#ff7f0e")
                    create_tooltip("Couleur d'accentuation pour les éléments secondaires")
                
                custom_css = f"""
                    .stApp {{ background-color: {background_color}; color: {text_color}; }}
                    .stButton>button {{ background-color: {primary_color}; color: white; }}
                    h1, h2, h3 {{ color: {primary_color}; }}
                    .sidebar .sidebar-content {{ background-color: {secondary_color}; }}
                """
                
                if st.button("Appliquer le thème"):
                    st.session_state.theme_settings.update({
                        'app_theme': 'custom', 'primary_color': primary_color,
                        'background_color': background_color, 'text_color': text_color
                    })
                    st.session_state.custom_css = custom_css
                    st.rerun()
        
        with st.expander("📊 Paramètres des Graphiques", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.graph_settings['width'] = st.slider("Largeur", 400, 1200, 800)
                create_tooltip("Largeur des graphiques en pixels")
                st.session_state.graph_settings['height'] = st.slider("Hauteur", 300, 900, 600)
                create_tooltip("Hauteur des graphiques en pixels")
            with col2:
                st.session_state.graph_settings['title_size'] = st.slider("Taille titre", 10, 24, 16)
                create_tooltip("Taille de police des titres des graphiques")
                st.session_state.graph_settings['axis_size'] = st.slider("Taille axes", 8, 20, 14)
                create_tooltip("Taille de police des labels d'axes")
            
            st.session_state.graph_settings['legend_size'] = st.slider("Taille légende", 8, 20, 12)
            create_tooltip("Taille de police des légendes")
            st.session_state.max_graphs_per_row = st.slider("Graphiques par ligne", 1, 4, 2)
            create_tooltip("Nombre maximum de graphiques affichés côte à côte")
            st.session_state.max_categories = st.slider("Max catégories", 5, 50, 20)
            create_tooltip("Nombre maximum de valeurs uniques pour qu'une variable soit considérée comme catégorielle")
        
        with st.expander("🔍 Gestion des Valeurs Manquantes", expanded=False):
            st.session_state.missing_threshold = st.slider(
                "Seuil de valeurs manquantes (%)", 0, 100, 60
            )
            create_tooltip("Pourcentage minimum de valeurs manquantes pour qu'une ligne soit considérée comme problématique")
            
            missing_action = st.radio(
                "Action sur les valeurs manquantes:",
                ["Afficher uniquement", "Exclure définitivement"]
            )
            create_tooltip("Choisissez si les lignes avec trop de valeurs manquantes doivent être exclues ou seulement signalées")
            
            st.session_state.missing_action = missing_action
            st.info(f"Les lignes avec plus de {st.session_state.missing_threshold}% de valeurs manquantes seront {missing_action.lower()}")
        
        st.markdown("---")
        st.header("📁 Chargement des Données")
        
        # CORRECTION: Simplifier la sélection de la source des données
        data_source = st.radio("Source des données", 
                              ["Fichiers locaux", "Datasets sklearn"],
                              key="data_source_radio",
                              horizontal=True)
        
        # Mettre à jour l'état de session
        st.session_state.data_source = data_source
        
        if st.session_state.data_source == "Fichiers locaux":
            # Sélecteur de séparateur
            separator_options = [ "automatique", "espace", ", (virgule)", "; (point-virgule)", "\t (tabulation)", "| (pipe)" ]
            selected_separator = st.selectbox("Séparateur de colonnes", separator_options)
            create_tooltip("Séparateur utilisé dans les fichiers CSV/TXT. La détection automatique analyse le fichier pour trouver le bon séparateur.")
            
            separator_map = {
                ", (virgule)": ",",
                "; (point-virgule)": ";", 
                "\t (tabulation)": "\t",
                "| (pipe)": "|",
                "espace": " ",
                "automatique": None
            }
            st.session_state.separator = separator_map[selected_separator]
            
            # Upload files
            uploaded_files = st.file_uploader("Sélectionnez les fichiers à analyser", 
                                            type=['csv', 'txt', 'xlsx', 'xls'], accept_multiple_files=True)
            create_tooltip("Chargez un ou plusieurs fichiers de données. Formats supportés: CSV, TXT, Excel (XLSX, XLS)")
            
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_files:
                    df = load_data(uploaded_file, st.session_state.separator)
                    if df is not None:
                        st.session_state.uploaded_files[uploaded_file.name] = df
                        st.session_state.original_data[uploaded_file.name] = df.copy()
                        st.session_state.filtered_data[uploaded_file.name] = df.copy()
                        st.session_state.data_loaded = True
                        st.success(f"✅ {uploaded_file.name} chargé ({len(df)} lignes, {len(df.columns)} colonnes)")
        
        else:  # Datasets sklearn
            dataset_options = ['Iris', 'Diabète', 'Digits', 'Linnerud', 'Wine', 'Breast Cancer', 'California Housing']
            selected_dataset = st.selectbox("Sélectionnez un dataset", dataset_options)
            
            # CORRECTION: Utiliser un formulaire pour éviter le rechargement immédiat
            with st.form(key="load_sklearn_dataset_form"):
                if st.form_submit_button("📥 Charger le dataset", type="primary"):
                    with st.spinner(f"Chargement du dataset {selected_dataset}..."):
                        df = load_sklearn_dataset(selected_dataset)
                        if df is not None:
                            dataset_name = f"sklearn_{selected_dataset.lower().replace(' ', '_')}"
                            st.session_state.uploaded_files[dataset_name] = df
                            st.session_state.original_data[dataset_name] = df.copy()
                            st.session_state.filtered_data[dataset_name] = df.copy()
                            st.session_state.data_loaded = True
                            st.session_state.sklearn_dataset = selected_dataset
                            st.session_state.sampling_file = dataset_name
                            st.success(f"✅ Dataset {selected_dataset} chargé ({len(df)} lignes, {len(df.columns)} colonnes)")
                            # Forcer le rechargement pour afficher les données
                            st.rerun()
        
        # CORRECTION: Afficher les fichiers chargés
        if st.session_state.uploaded_files:
            st.subheader("📂 Fichiers chargés")
            for file_name in st.session_state.uploaded_files.keys():
                df = st.session_state.uploaded_files[file_name]
                st.write(f"• {file_name} ({len(df)} lignes, {len(df.columns)} colonnes)")
        
        # CORRECTION: Sélection du fichier principal après chargement
        if st.session_state.uploaded_files:
            file_names = list(st.session_state.uploaded_files.keys())
            st.markdown("---")
            sampling_file = st.selectbox("📋 Fichier principal", file_names, 
                                       key="sampling_file_selector")
            
            if sampling_file:
                st.session_state.sampling_file = sampling_file
                df = st.session_state.uploaded_files[sampling_file]
                
                # Afficher les informations du fichier sélectionné
                st.info(f"**{sampling_file}** sélectionné: {len(df)} lignes, {len(df.columns)} colonnes")
                
                id_options = df.columns.tolist()
                st.session_state.id_column = st.selectbox("🔑 Colonne ID", id_options, index=0)
                create_tooltip("Colonne contenant les identifiants uniques des enregistrements")
        
        if st.session_state.uploaded_files and st.session_state.all_columns:
            st.markdown("---")
            with st.expander("🔍 Filtres de Base", expanded=True):
                st.subheader("Filtres par Colonne")
                
                filterable_cols = st.multiselect("Colonnes à filtrer", st.session_state.all_columns)
                create_tooltip("Sélectionnez les colonnes sur lesquelles appliquer des filtres")
                
                for col in filterable_cols:
                    col_type = "numeric"
                    for df in st.session_state.uploaded_files.values():
                        if col in df.columns:
                            if not pd.api.types.is_numeric_dtype(df[col]):
                                col_type = "categorical"
                                break
                            if df[col].nunique() <= st.session_state.max_categories:
                                col_type = "categorical"
                                break
                    
                    if col_type == "numeric":
                        min_val = float('inf')
                        max_val = float('-inf')
                        
                        for df in st.session_state.uploaded_files.values():
                            if col in df.columns:
                                col_min = df[col].min()
                                col_max = df[col].max()
                                
                                if not pd.isna(col_min) and col_min < min_val:
                                    min_val = col_min
                                if not pd.isna(col_max) and col_max > max_val:
                                    max_val = col_max
                        
                        if min_val == float('inf'):
                            min_val = 0
                        if max_val == float('-inf'):
                            max_val = 1
                            
                        selected_range = st.slider(
                            f"Plage pour {col}",
                            min_val, max_val, (min_val, max_val),
                            key=f"range_{col}"
                        )
                        create_tooltip(f"Définissez la plage de valeurs à conserver pour la colonne {col}")
                        st.session_state.filters[col] = ('range', selected_range)
                    
                    elif col_type == "categorical":
                        unique_vals = set()
                        for df in st.session_state.uploaded_files.values():
                            if col in df.columns:
                                unique_vals.update(df[col].unique().tolist())
                        
                        unique_vals = sorted([v for v in unique_vals if not pd.isna(v)])
                        selected_vals = st.multiselect(
                            f"Valeurs pour {col}",
                            unique_vals,
                            default=unique_vals,
                            key=f"cat_{col}"
                        )
                        create_tooltip(f"Sélectionnez les catégories à conserver pour la colonne {col}")
                        st.session_state.filters[col] = ('categories', selected_vals)
        
        if st.session_state.uploaded_files and st.session_state.all_columns:
            with st.expander("🔧 Filtres Avancés", expanded=False):
                st.subheader("Type de Filtre Avancé")
                
                st.session_state.filter_mode = st.radio(
                    "Sélectionnez le type de filtre à utiliser:",
                    ["Prédéfini", "Personnalisé"],
                    index=0
                )
                create_tooltip("Choisissez entre des filtres prédéfinis (différence, ratio, etc.) ou des filtres personnalisés avec expressions")
                
                if st.session_state.filter_mode == "Prédéfini":
                    st.subheader("Filtres Prédéfinis")
                    
                    predefined_filters = {
                        "Différence absolue": {
                            "expression": "abs(col1 - col2) > threshold",
                            "description": "Différence absolue entre deux colonnes"
                        },
                        "Ratio": {
                            "expression": "col1 / col2 > threshold",
                            "description": "Ratio entre deux colonnes"
                        },
                        "Somme": {
                            "expression": "col1 + col2 > threshold", 
                            "description": "Somme de deux colonnes"
                        },
                        "Produit": {
                            "expression": "col1 * col2 < threshold",
                            "description": "Produit de deux colonnes"
                        }
                    }
                    
                    selected_filter = st.selectbox("Sélectionnez un filtre prédéfini", 
                                                 list(predefined_filters.keys()))
                    create_tooltip("Choisissez un type de filtre prédéfini selon votre besoin d'analyse")
                    
                    if selected_filter:
                        st.write(f"**Description:** {predefined_filters[selected_filter]['description']}")
                        st.code(predefined_filters[selected_filter]['expression'])
                        
                        threshold_value = st.number_input("Valeur du seuil", value=0.1, step=0.1)
                        create_tooltip("Seuil à appliquer pour le filtre sélectionné")
                        
                        st.subheader("Mapping des Colonnes")
                        col_mapping = {}
                        col1_mapping = st.selectbox("Colonne 1 (col1)", [""] + st.session_state.all_columns)
                        create_tooltip("Sélectionnez la première colonne pour le filtre")
                        col2_mapping = st.selectbox("Colonne 2 (col2)", [""] + st.session_state.all_columns)
                        create_tooltip("Sélectionnez la deuxième colonne pour le filtre")
                        
                        if col1_mapping:
                            col_mapping['col1'] = col1_mapping
                        if col2_mapping:
                            col_mapping['col2'] = col2_mapping
                        
                        filter_name = st.text_input("Nom du filtre", f"filtre_{selected_filter.lower()}")
                        create_tooltip("Donnez un nom significatif à votre filtre pour le retrouver facilement")
                        
                        if st.button("➕ Ajouter le filtre prédéfini"):
                            if filter_name and col1_mapping and col2_mapping:
                                st.session_state.advanced_filters[filter_name] = {
                                    'expression': predefined_filters[selected_filter]['expression'],
                                    'column_mapping': col_mapping,
                                    'type': 'predefined',
                                    'threshold': threshold_value
                                }
                                st.success(f"Filtre '{filter_name}' ajouté")
                
                else:
                    st.subheader("Filtre Personnalisé")
                    
                    filter_name = st.text_input("Nom du filtre", "mon_filtre_personnalise")
                    create_tooltip("Nom unique pour identifier votre filtre personnalisé")
                    filter_expression = st.text_area("Expression du filtre", 
                                                   "abs(col1 - col2) < 0.1")
                    create_tooltip("Expression Python valide utilisant les noms de colonnes génériques (col1, col2, etc.)")
                    
                    st.subheader("Mapping des Colonnes")
                    col_mapping = {}
                    num_columns = st.slider("Nombre de colonnes à utiliser", 1, 10, 2)
                    create_tooltip("Nombre de colonnes à inclure dans votre filtre personnalisé")
                    
                    for i in range(1, num_columns + 1):
                        col_mapping[f'col{i}'] = st.selectbox(
                            f"Colonne {i}", 
                            [""] + st.session_state.all_columns
                        )
                        create_tooltip(f"Sélectionnez la colonne réelle correspondant à col{i}")
                    
                    if st.button("➕ Ajouter le filtre personnalisé"):
                        if filter_name and filter_expression:
                            st.session_state.advanced_filters[filter_name] = {
                                'expression': filter_expression,
                                'column_mapping': col_mapping,
                                'type': 'custom'
                            }
                            st.success(f"Filtre '{filter_name}' ajouté")
                
                if st.session_state.advanced_filters:
                    st.subheader("Filtres Actifs")
                    for name, info in list(st.session_state.advanced_filters.items()):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(f"{name}: {info['expression']}")
                            st.text(f"Mapping: {info['column_mapping']}")
                            if info.get('type') == 'predefined':
                                st.text(f"Seuil: {info.get('threshold', 'N/A')}")
                        with col2:
                            if st.button("🗑️", key=f"del_{name}"):
                                del st.session_state.advanced_filters[name]
                                st.rerun()
        
        if st.session_state.uploaded_files and st.session_state.all_columns:
            st.markdown("---")
            with st.expander("🔥 Configuration des Heatmaps", expanded=False):
                st.subheader("Sélection des Colonnes pour les Heatmaps")
                
                numeric_cols = []
                for df in st.session_state.uploaded_files.values():
                    numeric_cols.extend(df.select_dtypes(include=[np.number]).columns.tolist())
                numeric_cols = sorted(list(set(numeric_cols)))
                
                heatmap_columns = st.multiselect(
                    "Colonnes à inclure dans les heatmaps",
                    numeric_cols,
                    default=numeric_cols[:min(15, len(numeric_cols))]
                )
                create_tooltip("Sélectionnez les colonnes numériques à inclure dans les matrices de corrélation")
                
                st.session_state.heatmap_columns = heatmap_columns
                st.info(f"{len(heatmap_columns)} colonnes sélectionnées pour les heatmaps")
        
        if st.session_state.uploaded_files:
            st.header("🎛️ Contrôles")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Réinitialiser les filtres"):
                    st.session_state.filters = {}
                    st.session_state.advanced_filters = {}
                    for file_name in st.session_state.original_data:
                        st.session_state.filtered_data[file_name] = st.session_state.original_data[file_name].copy()
                    st.rerun()
            with col2:
                if st.button("📊 Appliquer les filtres"):
                    for file_name in st.session_state.original_data:
                        original_df = st.session_state.original_data[file_name]
                        filtered_df = apply_basic_filters(original_df, st.session_state.filters)
                        
                        if st.session_state.advanced_filters:
                            active_filters = {}
                            for name, info in st.session_state.advanced_filters.items():
                                if (st.session_state.filter_mode == "Prédéfini" and info.get('type') == 'predefined') or \
                                   (st.session_state.filter_mode == "Personnalisé" and info.get('type') == 'custom'):
                                    active_filters[name] = info
                            
                            filtered_df = apply_advanced_filters(filtered_df, active_filters, {})
                        
                        st.session_state.filtered_data[file_name] = filtered_df
                    st.rerun()
            with col3:
                if st.button("💾 Sauvegarder l'état"):
                    st.success("État de l'application sauvegardé")

def create_advanced_clustering_interface(df):
    """Crée l'interface pour le clustering avancé avec fonctionnalités d'ensemble"""
    st.header("🔍 Clustering Avancé")
    
    # Sélection des colonnes pour le clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("❌ Le clustering nécessite au moins 2 colonnes numériques")
        return
    
    st.subheader("📊 Sélection des Données")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_cols = st.multiselect(
            "Colonnes pour le clustering",
            numeric_cols,
            default=numeric_cols[:min(50, len(numeric_cols))],
            help="Sélectionnez les colonnes numériques à utiliser pour le clustering"
        )
        create_tooltip("Choisissez les variables numériques qui serviront de base au clustering")
    
    with col2:
        n_clusters = st.slider(
            "Nombre de clusters",
            min_value=2, max_value=20, value=3,
            help="Nombre de groupes à créer"
        )
        create_tooltip("Nombre de clusters (groupes) que vous souhaitez identifier dans vos données")
    
    # Sélection de la méthode de clustering
    st.subheader("⚙️ Méthode de Clustering")
    
    clustering_methods = {
        "KMeans": "Méthode des k-moyennes - Rapide et efficace pour des clusters sphériques",
        "DBSCAN": "Density-Based Spatial Clustering - Détecte les clusters de forme arbitraire",
        "Agglomerative": "Clustering hiérarchique agglomératif - Crée une hiérarchie de clusters",
        "Spectral": "Clustering spectral - Bon pour les données non convexes",
        "Gaussian Mixture": "Mélange de gaussiennes - Modèle probabiliste",
        "Ensemble": "Combinaison de plusieurs méthodes - Approche par consensus"
    }
    
    selected_method = st.selectbox(
        "Méthode de clustering",
        list(clustering_methods.keys()),
        help=clustering_methods[selected_method] if 'selected_method' in locals() else ""
    )
    create_tooltip(clustering_methods[selected_method] if selected_method in clustering_methods else "Choisissez l'algorithme de clustering adapté à vos données")
    
    # Paramètres spécifiques à la méthode
    method_params = {}
    if selected_method == "DBSCAN":
        col1, col2 = st.columns(2)
        with col1:
            method_params['eps'] = st.number_input("EPS (distance maximale)", value=0.5, step=0.1)
            create_tooltip("Distance maximale entre deux points pour qu'ils soient considérés comme voisins")
        with col2:
            method_params['min_samples'] = st.number_input("Échantillons minimum", value=5, min_value=1)
            create_tooltip("Nombre minimum de points requis pour former un cluster dense")
    elif selected_method == "Agglomerative":
        method_params['linkage'] = st.selectbox(
            "Lien", 
            ["ward", "complete", "average", "single"],
            help="Méthode de liaison pour le clustering hiérarchique"
        )
        create_tooltip("Critère de liaison utilisé pour mesurer la distance entre clusters")
    elif selected_method == "Ensemble":
        st.info("🔗 L'approche ensemble combine plusieurs méthodes pour un résultat plus robuste")
        
        ensemble_methods = st.multiselect(
            "Méthodes à inclure dans l'ensemble",
            ["KMeans", "DBSCAN", "Agglomerative", "Spectral", "Gaussian Mixture"],
            default=["KMeans", "Agglomerative", "Gaussian Mixture"]
        )
        create_tooltip("Sélectionnez les méthodes de clustering à combiner pour une approche consensus")
        
        method_params['ensemble_methods'] = ensemble_methods
    
    # Colonne de référence pour la comparaison
    st.subheader("📈 Comparaison avec une Colonne de Référence")
    
    reference_col = st.selectbox(
        "Colonne de référence (optionnel)",
        [""] + df.columns.tolist(),
        help="Comparez les clusters avec cette colonne pour évaluer la correspondance"
    )
    create_tooltip("Colonne existante permettant de valider la qualité du clustering (vérité terrain)")
    
    if reference_col == "":
        reference_col = None
    
    # Visualisation
    st.subheader("👁️ Visualisation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("Axe X", selected_cols, index=0)
        create_tooltip("Variable à afficher sur l'axe X de la visualisation")
    with col2:
        y_axis = st.selectbox("Axe Y", selected_cols, index=min(1, len(selected_cols)-1))
        create_tooltip("Variable à afficher sur l'axe Y de la visualisation")
    with col3:
        use_3d = st.checkbox("Visualisation 3D")
        create_tooltip("Active la visualisation en 3 dimensions")
        z_axis = None
        if use_3d and len(selected_cols) >= 3:
            z_axis = st.selectbox("Axe Z", selected_cols, index=min(2, len(selected_cols)-1))
            create_tooltip("Variable à afficher sur l'axe Z de la visualisation 3D")
    
    # Bouton d'exécution
    if st.button("🚀 Exécuter le Clustering", type="primary"):
        if len(selected_cols) < 2:
            st.error("❌ Sélectionnez au moins 2 colonnes pour le clustering")
            return
        
        with st.spinner("Clustering en cours..."):
            # Préparer les données
            clustering_data = df[selected_cols].dropna()
            
            if len(clustering_data) < n_clusters:
                st.error(f"❌ Pas assez de données valides ({len(clustering_data)}) pour créer {n_clusters} clusters")
                return
            
            # Appliquer le clustering
            try:
                if selected_method == "Ensemble":
                    # Clustering d'ensemble
                    methods_config = {}
                    for method in method_params['ensemble_methods']:
                        methods_config[method] = {'n_clusters': n_clusters}
                    
                    clusters, individual_clusterings, method_names, cooccurrence_matrix = ensemble_clustering(
                        clustering_data, methods_config, n_clusters
                    )
                    
                    st.session_state.clustering_results = {
                        'clusters': clusters,
                        'individual_clusterings': individual_clusterings,
                        'method_names': method_names,
                        'cooccurrence_matrix': cooccurrence_matrix,
                        'data': clustering_data,
                        'method': 'Ensemble',
                        'columns': selected_cols
                    }
                    
                else:
                    # Clustering simple
                    clusters, model, scaler = apply_clustering_method(
                        clustering_data, selected_method, n_clusters, **method_params
                    )
                    
                    # Calculer les métriques de qualité
                    metrics = calculate_cluster_metrics(clustering_data, clusters)
                    
                    # Stocker les résultats
                    st.session_state.clustering_results = {
                        'clusters': clusters,
                        'data': clustering_data,
                        'model': model,
                        'scaler': scaler,
                        'metrics': metrics,
                        'method': selected_method,
                        'columns': selected_cols
                    }
                
                # Comparaison avec la colonne de référence
                if reference_col is not None:
                    comparison_data = df.loc[clustering_data.index, reference_col]
                    cluster_comparison = calculate_cluster_matching(clusters, comparison_data)
                    st.session_state.cluster_comparison_results = cluster_comparison
                
                st.success("✅ Clustering terminé avec succès")
                
            except Exception as e:
                st.error(f"❌ Erreur lors du clustering: {str(e)}")
                return
        
        # Affichage des résultats
        st.subheader("📊 Résultats du Clustering")
        
        if selected_method == "Ensemble":
            # Résultats de l'approche ensemble
            st.write("**Approche Ensemble - Méthodes combinées:**")
            for i, method in enumerate(st.session_state.clustering_results['method_names']):
                st.write(f"- {method}")
            
            # Matrice d'accord entre méthodes
            agreement_matrix = calculate_cluster_agreement_matrix(
                st.session_state.clustering_results['individual_clusterings']
            )
            
            if agreement_matrix.size > 0:
                fig = px.imshow(
                    agreement_matrix,
                    x=st.session_state.clustering_results['method_names'],
                    y=st.session_state.clustering_results['method_names'],
                    title="Accord entre méthodes de clustering (ARI)",
                    color_continuous_scale="Blues",
                    text_auto=True
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.warning("Impossible de calculer la matrice d'accord")
            
        else:
            # Métriques de qualité pour les méthodes simples
            metrics = st.session_state.clustering_results['metrics']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score de silhouette", f"{metrics['silhouette']:.3f}" if not np.isnan(metrics['silhouette']) else "N/A")
            with col2:
                st.metric("Score Calinski-Harabasz", f"{metrics['calinski_harabasz']:.1f}" if not np.isnan(metrics['calinski_harabasz']) else "N/A")
            with col3:
                st.metric("Score Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}" if not np.isnan(metrics['davies_bouldin']) else "N/A")
        
        # Visualisation
        if use_3d and z_axis:
            fig = create_3d_cluster_plot(
                df.loc[clustering_data.index], x_axis, y_axis, z_axis, 
                st.session_state.clustering_results['clusters']
            )
        else:
            fig = px.scatter(
                df.loc[clustering_data.index], x=x_axis, y=y_axis, 
                color=st.session_state.clustering_results['clusters'].astype(str),
                title=f"Clustering {selected_method} - {x_axis} vs {y_axis}",
                labels={'color': 'Cluster'}
            )
        
        st.plotly_chart(fig, width='stretch')
        
        # Comparaison avec la colonne de référence
        if reference_col is not None and st.session_state.cluster_comparison_results:
            st.subheader("📈 Comparaison avec la Colonne de Référence")
            
            comparison = st.session_state.cluster_comparison_results
            
            if comparison['matching_type'] == 'categorical':
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score ARI", f"{comparison['adjusted_rand_score']:.3f}")
                with col2:
                    st.metric("Score NMI", f"{comparison['normalized_mutual_info']:.3f}")
                
                # Heatmap de correspondance
                heatmap_fig = create_cluster_comparison_heatmap(
                    comparison['iou_matrix'],
                    comparison['unique_clusters'],
                    comparison['unique_categories']
                )
                st.plotly_chart(heatmap_fig, width='stretch')
                
                # Meilleure correspondance par catégorie
                st.write("**Meilleure correspondance par catégorie:**")
                best_matches = []
                for j, category in enumerate(comparison['unique_categories']):
                    best_cluster_idx = np.argmax(comparison['iou_matrix'][:, j])
                    best_iou = comparison['iou_matrix'][best_cluster_idx, j]
                    best_matches.append({
                        'Catégorie': category,
                        'Meilleur Cluster': f"Cluster {comparison['unique_clusters'][best_cluster_idx]}",
                        'Score IoU': f"{best_iou:.3f}"
                    })
                
                st.dataframe(pd.DataFrame(best_matches), width='stretch')
            
            else:
                st.metric("Corrélation globale", f"{comparison['overall_correlation']:.3f}")
                st.write("**Moyennes par cluster:**")
                cluster_means_df = pd.DataFrame({
                    'Cluster': np.unique(st.session_state.clustering_results['clusters']),
                    f'Moyenne {reference_col}': comparison['cluster_means']
                })
                st.dataframe(cluster_means_df, width='stretch')
        
        # Statistiques par cluster
        st.subheader("📋 Statistiques par Cluster")
        
        clustered_df = df.loc[clustering_data.index].copy()
        clustered_df['Cluster'] = st.session_state.clustering_results['clusters']
        
        cluster_stats = clustered_df.groupby('Cluster')[selected_cols].agg(['mean', 'std', 'count'])
        st.dataframe(cluster_stats, width='stretch')
        
        # Téléchargement des résultats
        st.subheader("💾 Export des Résultats")
        
        csv_data = clustered_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les données clusterisées (CSV)",
            data=csv_data,
            file_name="donnees_clusterisees.csv",
            mime="text/csv"
        )

def create_advanced_multivariate_analysis(df):
    """Crée l'interface pour l'analyse multivariée avancée"""
    st.header("📊 Analyse Multivariée Avancée")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 3:
        st.warning("❌ L'analyse multivariée nécessite au moins 3 colonnes numériques")
        return
    
    st.subheader("🔧 Configuration de l'Analyse")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_cols = st.multiselect(
            "Colonnes numériques pour l'analyse",
            numeric_cols,
            default=numeric_cols[:min(50, len(numeric_cols))],
            help="Sélectionnez au moins 3 colonnes numériques"
        )
        create_tooltip("Variables numériques à inclure dans l'analyse en composantes principales")
    
    with col2:
        color_col = st.selectbox(
            "Colonne pour la coloration",
            [""] + df.columns.tolist(),
            help="Colonne utilisée pour colorer les points sur le graphique PCA"
        )
        create_tooltip("Variable catégorielle pour colorer les points dans la visualisation PCA")
        if color_col == "":
            color_col = None
    
    with col3:
        clustering_method = st.selectbox(
            "Méthode de clustering sur les composantes PCA",
            ["Aucun", "KMeans", "Agglomerative", "Gaussian Mixture"],
            help="Applique un clustering sur les données transformées par PCA"
        )
        create_tooltip("Algorithme de clustering à appliquer sur les composantes principales")
        
        n_clusters = 3
        if clustering_method != "Aucun":
            n_clusters = st.slider("Nombre de clusters PCA", 2, 30, 3)
            create_tooltip("Nombre de clusters à identifier dans l'espace des composantes principales")
    
    # CORRECTION: Définir pca_method avec une valeur par défaut
    pca_method = st.radio("Méthode de réduction PCA", 
                         ["Récursive (recommandée)", "Directe"], 
                         horizontal=True,
                         help="Méthode récursive: réduit progressivement les dimensions. Méthode directe: réduit en une seule étape.")
    
    # Métrique de correspondance
    correspondence_metric = st.selectbox(
        "Métrique de correspondance",
        ["iou", "ari", "nmi"],
        help="Métrique pour comparer les clusters avant et après PCA"
    )
    create_tooltip("Métrique utilisée pour évaluer la similarité entre les clusters originaux et PCA")
    
    if st.button("🚀 Lancer l'analyse multivariée", type="primary"):
        if len(selected_cols) < 3:
            st.error("❌ Sélectionnez au moins 3 colonnes numériques")
            return
        
        with st.spinner("Analyse multivariée en cours..."):
            # Clustering avant PCA
            clustering_data = df[selected_cols].dropna()
            
            if len(clustering_data) < 10:
                st.error("❌ Pas assez de données valides pour l'analyse")
                return
            
            # Clustering sur les données originales
            original_clusters, _, _ = apply_clustering_method(
                clustering_data, "KMeans", n_clusters=n_clusters
            )
            
            # Analyse multivariée avec PCA
            pca_method_param = "récursive" if pca_method == "Récursive (recommandée)" else "directe"
            pca_df, variance_ratio, pca_clusters, cluster_metrics = advanced_multivariate_analysis(
                df, selected_cols, color_col, clustering_method, n_clusters, pca_method_param
            )
            
            if pca_df is None:
                st.error("❌ Erreur lors de l'analyse multivariée")
                return
            
            # Affichage des résultats
            st.subheader("📈 Résultats de l'Analyse en Composantes Principales")
            
            # Afficher les variances expliquées selon le nombre de composantes
            n_components = len(variance_ratio)
            cols = st.columns(min(5, n_components+1))
            for i in range(n_components):
                with cols[i]:
                    st.metric(f"Variance expliquée PC{i+1}", f"{variance_ratio[i]:.1%}")
            
            if n_components > 1:
                with cols[-1] :
                    st.metric("Variance expliquée Tot", f"{sum(variance_ratio):.1%}")

            # Visualisation PCA
            if n_components >= 3:
                # Visualisation 3D
                if color_col:
                    fig = px.scatter_3d(
                        pca_df, x='PC1', y='PC2', z='PC3', color=color_col,
                        title=f"PCA 3D - Coloré par {color_col}"
                    )
                else:
                    fig = px.scatter_3d(
                        pca_df, x='PC1', y='PC2', z='PC3',
                        title="Analyse en Composantes Principales (3D)"
                    )
            else:
                # Visualisation 2D
                if color_col:
                    fig = px.scatter(
                        pca_df, x='PC1', y='PC2', color=color_col,
                        title=f"PCA 2D - Coloré par {color_col}"
                    )
                else:
                    fig = px.scatter(
                        pca_df, x='PC1', y='PC2',
                        title="Analyse en Composantes Principales (2D)"
                    )
            
            st.plotly_chart(fig, width='stretch')
            
            # Résultats du clustering sur PCA
            if pca_clusters is not None and clustering_method != "Aucun":
                st.subheader("🔍 Résultats du Clustering sur les Composantes PCA")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score de silhouette", f"{cluster_metrics['silhouette']:.3f}")
                with col2:
                    st.metric("Score Calinski-Harabasz", f"{cluster_metrics['calinski_harabasz']:.1f}")
                with col3:
                    st.metric("Score Davies-Bouldin", f"{cluster_metrics['davies_bouldin']:.3f}")
                
                # Comparaison avec le clustering original
                if len(original_clusters) == len(pca_clusters):
                    correspondence_score = calculate_cluster_correspondence(
                        original_clusters, pca_clusters, correspondence_metric
                    )
                    
                    st.metric(
                        f"Correspondance clustering original/PCA ({correspondence_metric.upper()})",
                        f"{correspondence_score:.3f}"
                    )
                    
                    # Visualisation des clusters PCA
                    if n_components >= 3:
                        fig_cluster = px.scatter_3d(
                            pca_df, x='PC1', y='PC2', z='PC3', color=pca_clusters.astype(str),
                            title="Clustering sur les Composantes PCA"
                        )
                    else:
                        fig_cluster = px.scatter(
                            pca_df, x='PC1', y='PC2', color=pca_clusters.astype(str),
                            title="Clustering sur les Composantes PCA"
                        )
                    st.plotly_chart(fig_cluster, width='stretch')
            
            # Téléchargement des résultats PCA
            st.subheader("💾 Export des Résultats PCA")
            
            csv_data = pca_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les données PCA (CSV)",
                data=csv_data,
                file_name="donnees_pca.csv",
                mime="text/csv"
            )

def create_main_interface():
    """Crée l'interface principale de l'application"""
    st.title("📊 Analyse Avancée des Résultats d'Entraînement IA")
    st.markdown("""
    **Application modulaire pour l'analyse approfondie des résultats d'entraînement de modèles d'IA.**
    Chargez vos fichiers, configurez les visualisations et explorez vos données.
    """)
    
    # CORRECTION: Meilleur affichage de l'état des données
    if not st.session_state.uploaded_files:
        st.info("💡 Veuillez charger des fichiers de données dans la sidebar pour commencer.")
        
        # Aide pour le chargement des datasets sklearn
        if st.session_state.data_source == "Datasets sklearn":
            st.info("""
            **Pour charger un dataset sklearn :**
            1. Sélectionnez "Datasets sklearn" dans la sidebar
            2. Choisissez un dataset dans la liste
            3. Cliquez sur "📥 Charger le dataset"
            
            **Datasets disponibles :**
            - **Iris** : Données de fleurs iris (classification)
            - **Diabète** : Données médicales sur le diabète (régression)
            - **Digits** : Images de chiffres manuscrits (classification)
            - **Linnerud** : Données d'exercice physique (régression multivariée)
            - **Wine** : Données chimiques de vins (classification)
            - **Breast Cancer** : Données sur le cancer du sein (classification)
            - **California Housing** : Données immobilières (régression)
            """)
        else:
            st.info("""
            **Pour charger vos propres fichiers :**
            1. Sélectionnez "Fichiers locaux" dans la sidebar
            2. Choisissez le type de séparateur si nécessaire
            3. Téléchargez vos fichiers CSV, TXT, Excel
            """)
        return
    
    # CORRECTION: Vérifier que le fichier principal est défini
    if not st.session_state.sampling_file:
        st.warning("⚠️ Veuillez sélectionner un fichier principal dans la sidebar")
        return
        
    if st.session_state.sampling_file and st.session_state.sampling_file in st.session_state.filtered_data:
        df = st.session_state.filtered_data[st.session_state.sampling_file]
    else:
        # Fallback: utiliser le premier fichier disponible
        df = st.session_state.uploaded_files[list(st.session_state.uploaded_files.keys())[0]]
        st.session_state.sampling_file = list(st.session_state.uploaded_files.keys())[0]
    
    # CORRECTION: Afficher des informations plus claires sur les données
    st.success(f"✅ **Données chargées :** {st.session_state.sampling_file}")
    
    # Mettre à jour toutes les colonnes disponibles
    all_columns = set()
    for df_temp in st.session_state.uploaded_files.values():
        all_columns.update(df_temp.columns.tolist())
    st.session_state.all_columns = sorted(list(all_columns))
    
    # Calcul des métriques avancées
    initial_rows = len(st.session_state.original_data.get(st.session_state.sampling_file, df))
    high_missing_rows, missing_percentage = calculate_missing_rows(df, st.session_state.missing_threshold)
    
    filtered_rows = len(df)
    removed_rows = initial_rows - filtered_rows
    removed_percentage = (removed_rows / initial_rows) * 100 if initial_rows > 0 else 0
    
    # Affichage des métriques
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    with col1:
        st.metric("Fichiers chargés", len(st.session_state.uploaded_files), border=True)
    with col2:
        st.metric("Enregistrements totaux", initial_rows, border=True)
    with col3:
        st.metric("Colonnes disponibles", len(st.session_state.all_columns), border=True)
    with col4:
        st.metric("Filtres actifs", len(st.session_state.filters) + len(st.session_state.advanced_filters), border=True)
    with col5:
        st.metric("Lignes supprimées", f"{removed_rows} ({removed_percentage:.1f}%)", border=True)
    with col6:
        st.metric("Lignes à valeurs manquantes", f"{high_missing_rows} ({missing_percentage:.1f}%)", border=True)
    
    # Affichage des données filtrées
    with st.expander("👀 Aperçu des Données Filtrées", expanded=False):
        st.dataframe(df.head(10), width='stretch')
        st.write(f"**{len(df)}** enregistrements après filtrage")
        
        # Afficher les types de données
        st.subheader("📋 Types de Données")
        type_info = []
        for col in df.columns:
            dtype = df[col].dtype
            unique = df[col].nunique()
            missing = df[col].isnull().sum()
            type_info.append({
                'Colonne': col,
                'Type': dtype,
                'Valeurs uniques': unique,
                'Valeurs manquantes': missing
            })
        st.dataframe(pd.DataFrame(type_info), width='stretch')
    
    # Onglets d'analyse
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Distribution", "🔗 Corrélations", "📊 Catégoriel", 
        "📋 Échantillonnage", "🔥 Heatmaps", "🔍 Clustering", "📈 Régression", "⚡ Avancé", "📚 Métriques"
    ])
    
    # Onglet Distribution
    with tab1:
        st.header("Analyse de Distribution")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()
            selected_col = st.selectbox("Colonne à analyser", all_cols, key="dist_col")
            create_tooltip("Sélectionnez la variable dont vous souhaitez étudier la distribution")
        with col2:
            plot_types = ["histogram", "box", "violin", "density", "bar", "pie"]
            plot_type = st.selectbox("Type de visualisation", plot_types, key="dist_type")
            create_tooltip("Choisissez le type de graphique le plus adapté à votre analyse")
        with col3:
            nbins = 30
            density = False
            y_col_option = None
            
            if plot_type == "histogram":
                nbins = st.slider("Nombre d'intervalles", 5, 500, 30, key="dist_nbins")
                create_tooltip("Nombre de barres dans l'histogramme")
                density = st.checkbox("Afficher la densité", key="dist_density")
                create_tooltip("Superpose une courbe de densité à l'histogramme")
            elif plot_type == "bar":
                y_col_option = st.selectbox("Colonne Y (optionnel)", [None] + all_cols, key="dist_y_col")
                create_tooltip("Variable quantitative pour un diagramme en barres groupé")
        
        col1, col2 = st.columns(2)
        with col1:
            color_col = st.selectbox("Couleur", [None] + all_cols, key="dist_color")
            create_tooltip("Variable catégorielle pour colorer les éléments du graphique")
        with col2:
            facet_col = st.selectbox("Facettes", [None] + all_cols, key="dist_facet")
            create_tooltip("Variable catégorielle pour créer des sous-graphiques multiples")
        
        if selected_col:
            fig = create_custom_plot(
                df, selected_col, y_col=y_col_option, plot_type=plot_type,
                color_col=color_col, facet_col=facet_col,
                theme_settings=st.session_state.theme_settings,
                fig_size=st.session_state.graph_settings,
                nbins=nbins, density=density
            )
            if fig:
                st.plotly_chart(fig, width='stretch', key=f"dist_chart_{selected_col}_{plot_type}")
    
    # Onglet Corrélations
    with tab2:
        st.header("Analyse des Corrélations")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()
            x_col = st.selectbox("Variable X", all_cols, key="corr_x")
            create_tooltip("Variable indépendante ou explicative")
        with col2:
            y_col = st.selectbox("Variable Y", all_cols, key="corr_y")
            create_tooltip("Variable dépendante ou à expliquer")
        with col3:
            use_3d_corr = st.checkbox("Visualisation 3D", key="corr_3d")
            create_tooltip("Active la visualisation tridimensionnelle")
            z_col = None
            if use_3d_corr:
                z_col = st.selectbox("Variable Z", all_cols, key="corr_z")
                create_tooltip("Troisième variable pour l'analyse 3D")
        
        if not use_3d_corr:
            corr_types = ["scatter", "line", "density", "pairplot"]
            corr_type = st.selectbox("Type de graphique", corr_types, key="corr_type")
            create_tooltip("Type de visualisation des relations entre variables")
        else:
            corr_type = "scatter"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            color_col = st.selectbox("Couleur pour corrélation", [None] + all_cols, key="corr_color")
            create_tooltip("Variable de coloration pour différencier les groupes")
        with col2:
            if not use_3d_corr:
                facet_col = st.selectbox("Facettes pour corrélation", [None] + all_cols, key="corr_facet")
                create_tooltip("Crée des sous-graphiques par catégorie")
        with col3:
            if corr_type == "scatter" and not use_3d_corr:
                trendline_options = [None, "ols", "lowess", "expanding", "rolling"]
                trendline = st.selectbox("Ligne de tendance", trendline_options, key="corr_trendline")
                create_tooltip("Ajoute une ligne de régression ou de tendance")
            else:
                trendline = None
        
        if x_col and y_col:
            dimensions = 3 if use_3d_corr and z_col else 2
            fig = create_custom_plot(
                df, x_col, y_col, z_col=z_col, plot_type=corr_type,
                color_col=color_col, facet_col=facet_col if not use_3d_corr else None, 
                trendline=trendline, dimensions=dimensions,
                theme_settings=st.session_state.theme_settings,
                fig_size=st.session_state.graph_settings
            )
            if fig:
                st.plotly_chart(fig, width='stretch', key=f"corr_chart_{x_col}_{y_col}_{corr_type}")
                
                if x_col in numeric_cols and y_col in numeric_cols and x_col != y_col and not use_3d_corr:
                    pearson_corr = df[x_col].corr(df[y_col])
                    brice_corrs = calculate_brice_correlation(df[x_col], df[y_col])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Corrélation Pearson", f"{pearson_corr:.3f}", border=True, chart_type="bar")
                    with col2:
                        st.metric("Corrélation Brice", f"{brice_corrs['brice']:.3f}", border=True, chart_type="bar")
                    with col3:
                        st.metric("Corrélation Brice1", f"{brice_corrs['brice1']:.3f}", border=True, chart_type="bar")
    
    # Onglet Catégoriel
    with tab3:
        st.header("Analyse des Données Catégorielles")
        
        categorical_cols = []
        for col in df.columns:
            if df[col].nunique() <= st.session_state.max_categories:
                categorical_cols.append(col)
        
        if not categorical_cols:
            st.warning("Aucune colonne catégorielle détectée. Augmentez le 'Max catégories' dans la sidebar.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                cat_col = st.selectbox("Colonne catégorielle", categorical_cols, key="cat_col")
                create_tooltip("Variable catégorielle à analyser")
            with col2:
                cat_plot_types = ["bar", "pie", "box", "violin", "histogram"]
                cat_plot_type = st.selectbox("Type de graphique", cat_plot_types, key="cat_plot_type")
                create_tooltip("Type de visualisation pour données catégorielles")
            with col3:
                value_col = st.selectbox("Colonne de valeurs", [None] + numeric_cols, key="cat_value")
                create_tooltip("Variable quantitative pour les graphiques à deux variables")
            
            col1, col2 = st.columns(2)
            with col1:
                color_col = st.selectbox("Couleur pour catégoriel", [None] + all_cols, key="cat_color")
                create_tooltip("Variable de coloration supplémentaire")
            with col2:
                facet_col = st.selectbox("Facettes pour catégoriel", [None] + all_cols, key="cat_facet")
                create_tooltip("Variable de segmentation pour sous-graphiques")
            
            if cat_col:
                if value_col and cat_plot_type in ["box", "violin"]:
                    if cat_plot_type == "box":
                        fig = px.box(df, x=cat_col, y=value_col, color=color_col, facet_col=facet_col,
                                   title=f"{value_col} par {cat_col}")
                    else:
                        fig = px.violin(df, x=cat_col, y=value_col, color=color_col, facet_col=facet_col,
                                     box=True, title=f"{value_col} par {cat_col}")
                else:
                    value_counts = df[cat_col].value_counts()
                    if cat_plot_type == "bar":
                        if color_col:
                            fig = px.histogram(df, x=cat_col, color=color_col, facet_col=facet_col,
                                            title=f"Distribution de {cat_col} par {color_col}")
                        else:
                            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                      title=f"Distribution de {cat_col}")
                    elif cat_plot_type == "pie":
                        fig = px.pie(values=value_counts.values, names=value_counts.index, 
                                   title=f"Distribution de {cat_col}")
                    elif cat_plot_type == "histogram":
                        fig = px.histogram(df, x=cat_col, color=color_col, facet_col=facet_col,
                                         title=f"Distribution de {cat_col}")
                
                fig.update_layout(
                    width=st.session_state.graph_settings['width'],
                    height=st.session_state.graph_settings['height'],
                    font_size=st.session_state.theme_settings['font_size']
                )
                
                st.plotly_chart(fig, width='stretch', key=f"cat_chart_{cat_col}_{cat_plot_type}")
                
                st.subheader("Statistiques Descriptives")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nombre de catégories", df[cat_col].nunique(), border=True, chart_type="bar")
                with col2:
                    st.metric("Valeur la plus fréquente", df[cat_col].mode().iloc[0] if not df[cat_col].mode().empty else "N/A", border=True, chart_type="bar")
                with col3:
                    st.metric("Valeurs manquantes", df[cat_col].isnull().sum(), border=True, chart_type="bar")
    
    # Onglet Échantillonnage
    with tab4:
        st.header("📋 Analyse des Données d'Échantillonnage")
        
        if st.session_state.sampling_file:
            df_sampling = st.session_state.filtered_data.get(st.session_state.sampling_file, 
                                                           st.session_state.uploaded_files[st.session_state.sampling_file])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                interval_prefix = st.text_input("Préfixe intervalles", "Interval_", key="int_prefix")
                create_tooltip("Préfixe des colonnes contenant les intervalles d'échantillonnage")
            with col2:
                stat_prefixes = st.text_input("Préfixes stats", "Ech_count_,Ech_mean_,Ech_std_,Ech_min_,Ech_max_", 
                                            key="stat_prefix")
                create_tooltip("Préfixes des colonnes de statistiques d'échantillonnage (séparés par des virgules)")
            with col3:
                num_prefix = st.text_input("Préfixe nombre", "Numb_intervals_", key="num_prefix")
                create_tooltip("Préfixe de la colonne indiquant le nombre d'intervalles")
            
            stat_list = [s.strip() for s in stat_prefixes.split(",") if s.strip()]
            
            if st.button("🔍 Analyser l'échantillonnage", type="primary"):
                with st.spinner("Analyse en cours..."):
                    st.session_state.analysis_results = analyze_sampling_data_improved(
                        df_sampling, interval_prefix, stat_list, num_prefix
                    )
            
            if st.session_state.analysis_results:
                st.success(f"✅ {len(st.session_state.analysis_results)} suffixes analysés")
                
                col1, col2 = st.columns(2)
                with col1:
                    available_suffixes = list(st.session_state.analysis_results.keys())
                    selected_suffixes = st.multiselect("Suffixes à visualiser", 
                                                     available_suffixes, 
                                                     default=available_suffixes[:min(2, len(available_suffixes))])
                    create_tooltip("Sélectionnez les groupes d'échantillonnage à visualiser")
                with col2:
                    available_stats = set()
                    for suffix in selected_suffixes:
                        if suffix in st.session_state.analysis_results:
                            available_stats.update(st.session_state.analysis_results[suffix]['data'].keys())
                    
                    selected_stats = st.multiselect("Statistiques à visualiser", 
                                                  list(available_stats),
                                                  default=list(available_stats)[:min(2, len(available_stats))])
                    create_tooltip("Types de statistiques à afficher (moyenne, écart-type, etc.)")
                
                viz_type = st.radio("Type de visualisation", ["line", "bar"], horizontal=True)
                create_tooltip("Choisissez entre un graphique linéaire ou en barres")
                
                if selected_suffixes and selected_stats:
                    fig = create_sampling_visualization(
                        st.session_state.analysis_results, selected_suffixes, selected_stats,
                        viz_type, st.session_state.theme_settings, st.session_state.graph_settings
                    )
                    
                    if fig:
                        st.plotly_chart(fig, width='stretch', key="sampling_chart")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            pdf_name = st.text_input("Nom du PDF", "analyse_echantillonnage.pdf")
                            create_tooltip("Nom du fichier PDF à générer")
                        with col2:
                            if st.button("💾 Exporter en PDF"):
                                pdf_data = save_plot_as_pdf(fig, pdf_name)
                                if pdf_data:
                                    st.download_button(
                                        label="📥 Télécharger PDF",
                                        data=pdf_data,
                                        file_name=pdf_name,
                                        mime="application/pdf"
                                    )
            else:
                st.info("👆 Cliquez sur 'Analyser l'échantillonnage' pour commencer l'analyse")
        else:
            st.warning("Veuillez sélectionner un fichier d'échantillonnage dans la sidebar")
    
    # Onglet Heatmaps
    with tab5:
        st.header("🔥 Analyse par Heatmaps")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.warning("❌ Aucune colonne numérique trouvée dans le dataset.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                heatmap_type = st.selectbox(
                    "Type de heatmap",
                    ["heatmap", "heatmap_brice", "heatmap_brice1"]
                )
                create_tooltip("Type de corrélation à calculer: Pearson (standard), Brice ou Brice1 (variations)")

            with col2:
                max_default_cols = st.slider("Nombre de colonnes par défaut", 5, 30, 15)
                create_tooltip("Nombre maximum de colonnes à inclure par défaut dans la heatmap")

            st.subheader("Sélection des colonnes numériques")

            if not numeric_cols:
                st.error("Aucune colonne numérique disponible.")
            else:
                default_selected = numeric_cols[:min(max_default_cols, len(numeric_cols))]

                selected_columns = st.multiselect(
                    "Colonnes à inclure dans la heatmap:",
                    options=numeric_cols,
                    default=default_selected
                )
                create_tooltip("Sélectionnez les variables numériques pour la matrice de corrélation")

                st.info(f"**{len(selected_columns)}** colonnes sélectionnées sur **{len(numeric_cols)}** disponibles")

            st.subheader("Options d'affichage")
            col1, col2, col3 = st.columns(3)
            with col1:
                show_values = st.checkbox("Afficher les valeurs", value=True)
                create_tooltip("Affiche les valeurs numériques dans les cellules de la heatmap")
            with col2:
                color_scale = st.selectbox("Échelle de couleurs", 
                                         ["RdBu_r", "Viridis", "Plasma", "Inferno", "Blues", "Reds"])
                create_tooltip("Palette de couleurs pour la visualisation")
            with col3:
                fig_width = st.slider("Largeur du graphique", 600, 1200, 800)
                create_tooltip("Largeur de la heatmap en pixels")

            if st.button("🔄 Générer la heatmap", type="primary"):
                if len(selected_columns) < 2:
                    st.error("❌ Sélectionnez au moins 2 colonnes numériques pour générer une heatmap.")
                else:
                    with st.spinner("Calcul de la matrice de corrélation..."):
                        try:
                            valid_columns = []
                            for col in selected_columns:
                                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                                    if df[col].notna().sum() >= 2:
                                        valid_columns.append(col)

                            if len(valid_columns) < 2:
                                st.error(f"❌ Seules {len(valid_columns)} colonnes valides.")
                            else:
                                fig = create_custom_plot(
                                    df, None, None, plot_type=heatmap_type,
                                    theme_settings=st.session_state.theme_settings,
                                    fig_size={'width': fig_width, 'height': 600},
                                    heatmap_columns=valid_columns,
                                    max_heatmap_cols=len(valid_columns)
                                )

                                if fig:
                                    fig.update_layout(coloraxis=dict(colorscale=color_scale))

                                    if not show_values:
                                        fig.update_traces(texttemplate='')

                                    st.plotly_chart(fig, width='stretch', key=f"{heatmap_type}_chart")

                                    st.success(f"✅ Heatmap générée avec {len(valid_columns)} colonnes numériques")

                                    with st.expander("📊 Statistiques des données utilisées"):
                                        st.write(f"**Colonnes utilisées:** {', '.join(valid_columns)}")
                                        st.write(f"**Nombre de lignes:** {len(df)}")
                                        st.write(f"**Valeurs manquantes:** {df[valid_columns].isnull().sum().sum()}")

                                    if heatmap_type == "heatmap":
                                        corr_matrix = df[valid_columns].corr()
                                        csv_corr = corr_matrix.to_csv()
                                        st.download_button(
                                            label="📥 Télécharger la matrice de corrélation (CSV)",
                                            data=csv_corr,
                                            file_name="matrice_correlation.csv",
                                            mime="text/csv"
                                        )
                                else:
                                    st.error("❌ Erreur lors de la génération de la heatmap.")

                        except Exception as e:
                            st.error(f"❌ Erreur lors de la création de la heatmap: {str(e)}")

            with st.expander("❓ Aide sur les heatmaps"):
                st.markdown("""
                ### Guide d'utilisation des heatmaps

                **1. Sélection des colonnes:**
                - Seules les colonnes numériques sont disponibles
                - Sélectionnez au moins 2 colonnes pour une heatmap valide

                **2. Types de heatmaps:**
                - **Heatmap standard**: Corrélation de Pearson (linéaire)
                - **Heatmap Brice**: Corrélation basée sur les variations
                - **Heatmap Brice1**: Corrélation basée sur les variations centrées

                **3. Interprétation des couleurs:**
                - 🔴 Rouge: Corrélation positive forte
                - 🔵 Bleu: Corrélation négative forte  
                - ⚪ Blanc: Peu ou pas de corrélation
                """)
    
    # Onglet Clustering Avancé
    with tab6:
        create_advanced_clustering_interface(df)
    
    # NOUVEL ONGLET: RÉGRESSION
    with tab7:
        create_regression_interface(df)
    
    # Onglet Avancé (avec optimisation d'hyperparamètres et analyse multivariée)
    with tab8:
        st.header("⚡ Outils d'Analyse Avancée")
        
        subtab1, subtab2, subtab3, subtab4, subtab5, subtab6 = st.tabs([
            "📈 Statistiques", "🔍 Outliers", "⏰ Séries Temporelles", 
            "📊 Analyse Multivariée", "🔎 Optimisation Hyperparamètres", "🎯 Ensemble Clustering"
        ])
        
        with subtab1:
            st.subheader("📈 Analyse Statistique Complète")
            
            if st.button("📊 Générer le rapport statistique complet"):
                with st.spinner("Génération du rapport..."):
                    st.write("### Statistiques Descriptives")
                    st.dataframe(df.describe(), width='stretch')
                    
                    st.write("### Types de Données")
                    type_info = []
                    for col in df.columns:
                        dtype = df[col].dtype
                        unique = df[col].nunique()
                        missing = df[col].isnull().sum()
                        type_info.append({
                            'Colonne': col,
                            'Type': dtype,
                            'Valeurs uniques': unique,
                            'Valeurs manquantes': missing
                        })
                    st.dataframe(pd.DataFrame(type_info), width='stretch')
                    
                    st.write("### Matrice de Corrélation (Pearson)")
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 1:
                        corr_matrix = numeric_df.corr()
                        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                      color_continuous_scale='RdBu_r',
                                      title="Matrice de Corrélation entre Variables Numériques")
                        st.plotly_chart(fig, width='stretch', key="advanced_corr_matrix")
                    
                    if len(numeric_df.columns) > 1:
                        st.write("### Top des Corrélations")
                        corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_pairs.append({
                                    'Variable 1': corr_matrix.columns[i],
                                    'Variable 2': corr_matrix.columns[j],
                                    'Corrélation': corr_matrix.iloc[i, j]
                                })
                        corr_df = pd.DataFrame(corr_pairs)
                        corr_df['Abs_Correlation'] = corr_df['Corrélation'].abs()
                        st.dataframe(corr_df.nlargest(10, 'Abs_Correlation'), width='stretch')
        
        with subtab2:
            st.subheader("🔍 Détection des Outliers")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Colonne à analyser", numeric_cols, key="outlier_col")
                create_tooltip("Variable numérique pour la détection des valeurs aberrantes")
                
                if selected_col:
                    outliers, lower_bound, upper_bound = detect_outliers_iqr(df, selected_col)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Outliers détectés", len(outliers), border=True, chart_type="bar")
                    with col2:
                        st.metric("Borne inférieure", f"{lower_bound:.2f}", border=True, chart_type="bar")
                    with col3:
                        st.metric("Borne supérieure", f"{upper_bound:.2f}", border=True, chart_type="bar")
                    
                    fig = px.box(df, y=selected_col, title=f"Distribution de {selected_col} avec outliers")
                    fig.add_hline(y=lower_bound, line_dash="dash", line_color="red", 
                                annotation_text="Borne inférieure")
                    fig.add_hline(y=upper_bound, line_dash="dash", line_color="red",
                                annotation_text="Borne supérieure")
                    st.plotly_chart(fig, width='stretch')
                    
                    if len(outliers) > 0:
                        st.write("### Détail des Outliers")
                        st.dataframe(outliers[[selected_col]], width='stretch')
            else:
                st.warning("Aucune colonne numérique disponible pour l'analyse des outliers")
        
        with subtab3:
            st.subheader("⏰ Analyse des Séries Temporelles")
            
            date_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col])
                        date_cols.append(col)
                    except:
                        pass
            
            if date_cols:
                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Colonne de date", date_cols, key="date_col")
                    create_tooltip("Colonne contenant les dates ou timestamps")
                with col2:
                    value_col = st.selectbox("Colonne de valeurs", numeric_cols, key="ts_value_col")
                    create_tooltip("Variable numérique à analyser dans le temps")
                
                if date_col and value_col:
                    try:
                        df_temp = df.copy()
                        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                        fig = create_time_series_analysis(df_temp, date_col, value_col)
                        st.plotly_chart(fig, width='stretch')
                    except Exception as e:
                        st.error(f"Erreur lors de la création de la série temporelle: {e}")
            else:
                st.info("ℹ️ Aucune colonne de date détectée.")
        
        with subtab4:
            create_advanced_multivariate_analysis(df)
        
        with subtab5:
            create_optimization_interface(df)
        
        with subtab6:
            st.header("🎯 Clustering d'Ensemble Avancé")
            st.info("""
            **Clustering d'Ensemble**: Cette approche combine plusieurs méthodes de clustering 
            pour obtenir un résultat plus robuste et fiable. Chaque méthode vote pour l'appartenance
            aux clusters, et un consensus est établi.
            """)
            
            # Cette fonctionnalité est déjà intégrée dans l'onglet Clustering principal
            st.success("✅ La fonctionnalité de clustering d'ensemble est disponible dans l'onglet '🔍 Clustering'")
            st.write("Sélectionnez 'Ensemble' comme méthode de clustering pour utiliser cette fonctionnalité.")
        
        # Section export des données
        st.markdown("---")
        st.subheader("💾 Export des Données")
        
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.radio("Format d'export", ["CSV", "Excel", "JSON"], horizontal=True)
            create_tooltip("Format de fichier pour l'export des données")
            export_name = st.text_input("Nom du fichier", "donnees_analyse")
            create_tooltip("Nom de base pour le fichier exporté")
        
        with col2:
            if st.button("📤 Exporter les données filtrées"):
                if export_format == "CSV":
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Télécharger CSV",
                        data=csv_data,
                        file_name=f"{export_name}.csv",
                        mime="text/csv"
                    )
                elif export_format == "Excel":
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Données')
                    st.download_button(
                        label="💾 Télécharger Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"{export_name}.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                elif export_format == "JSON":
                    json_data = df.to_json(indent=2, orient='records')
                    st.download_button(
                        label="💾 Télécharger JSON",
                        data=json_data,
                        file_name=f"{export_name}.json",
                        mime="application/json"
                    )
    
    # ONGLET: DICTIONNAIRE DES MÉTRIQUES
    with tab9:
        create_metrics_dictionary()

def main():
    """Fonction principale de l'application"""
    
    # Initialisation de l'état de session
    initialize_session_state()
    
    # Application du CSS personnalisé
    if st.session_state.custom_css:
        apply_custom_css(st.session_state.custom_css)
    
    # Création de l'interface
    create_sidebar()
    create_main_interface()
    
    # Pied de page
    st.markdown("---")
    st.markdown("**Application développée pour l'analyse avancée des résultats d'entraînement IA**")
    st.markdown("**BRICE KENGNI ZANGUIM** • *Dernière mise à jour: 2025-09-23*")

if __name__ == "__main__":
    main()
