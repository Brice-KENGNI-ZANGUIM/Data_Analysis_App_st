# =============================================================================
# IMPORTATION DES MODULES UTILITAIRES
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
    
    
def load_data(uploaded_file):
    """Charge un fichier de données avec gestion robuste des erreurs"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            # Essayer différents séparateurs
            content = uploaded_file.getvalue().decode('utf-8')
            if '\t' in content.split('\n')[0]:
                return pd.read_csv(uploaded_file, sep='\t')
            else:
                return pd.read_csv(uploaded_file, sep='\s+', engine='python')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            st.error(f"Format de fichier non supporté: {uploaded_file.name}")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement de {uploaded_file.name}: {e}")
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
    """Calcule les corrélations Brice et Brice1"""
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
        
        # --- Corrélation Brice : sur les variations (dérivées discrètes) ---
        x1 = x[1:] - x[:-1]
        y1 = y[1:] - y[:-1]
        
        # Écarts-types des variations
        s1 = np.sqrt(np.nanmean(x1 * x1))
        s2 = np.sqrt(np.nanmean(y1 * y1))
        
        brice_corr = np.nanmean(x1 * y1) / (s1 * s2) if (s1 * s2) != 0 else np.nan
        
        # --- Corrélation Brice1 : sur les variations centrées ---
        x1_centered = x1 - np.nanmean(x1)
        y1_centered = y1 - np.nanmean(y1)
        
        s1_centered = np.sqrt(np.nanmean(x1_centered * x1_centered))
        s2_centered = np.sqrt(np.nanmean(y1_centered * y1_centered))
        
        brice1_corr = np.nanmean(x1_centered * y1_centered) / (s1_centered * s2_centered) if (s1_centered * s2_centered) != 0 else np.nan
        
        return {'brice': brice_corr, 'brice1': brice1_corr}
    
    except Exception as e:
        return {'brice': np.nan, 'brice1': np.nan}

def create_custom_plot(data, x_col, y_col=None, plot_type="histogram", 
                      color_col=None, facet_col=None, theme_settings=None,
                      custom_colors=None, fig_size=None, font_sizes=None,
                      nbins=30, line_width=2, density=False, trendline=None,
                      heatmap_columns=None, max_heatmap_cols=15):
    """Crée des graphiques personnalisés avec une grande flexibilité"""
    
    if theme_settings is None:
        theme_settings = {}
    if custom_colors is None:
        custom_colors = px.colors.qualitative.Plotly
    if fig_size is None:
        fig_size = {'width': 800, 'height': 600}
    if font_sizes is None:
        font_sizes = {'title': 16, 'axis': 14, 'legend': 12}
    
    try:
        # CORRECTION : Séparer d'abord les heatmaps des autres graphiques
        if plot_type in ["heatmap", "heatmap_brice", "heatmap_brice1"]:
            # =============================================================================
            # SECTION SPÉCIALISÉE POUR LES HEATMAPS
            # =============================================================================
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
        
        # =============================================================================
        # SECTION POUR LES AUTRES TYPES DE GRAPHIQUES (EXISTANTE)
        # =============================================================================
        
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
                fig = px.density_contour(data, x=x_col, y=y_col, color=color_col,
                                        title=f"Densité de {x_col} vs {y_col}",
                                        color_discrete_sequence=custom_colors)
            
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
        
        # Application des paramètres de thème pour les autres graphiques
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

def create_cluster_analysis(df, columns, n_clusters=3):
    """Crée une analyse de clustering simple"""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Préparer les données
    data = df[columns].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Appliquer K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Ajouter les clusters au dataframe
    result_df = data.copy()
    result_df['Cluster'] = clusters
    
    # Visualiser avec les deux premières colonnes
    if len(columns) >= 2:
        fig = px.scatter(result_df, x=columns[0], y=columns[1], color='Cluster',
                        title=f"Clustering des données (K={n_clusters})")
        return fig, result_df
    
    return None, result_df

# =============================================================================
# INTERFACE STREAMLIT AMÉLIORÉE
# =============================================================================

def initialize_session_state():
    """Initialise l'état de session avec toutes les variables nécessaires"""
    defaults = {
        'uploaded_files': {},
        'original_data': {},  # Stocke les données originales (non modifiées)
        'filtered_data': {},   # Stocke les données après filtrage
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
        'all_columns': [],  # Stocke toutes les colonnes de tous les fichiers
        'missing_threshold': 60,  # Seuil pour les valeurs manquantes (en pourcentage)
        'filter_mode': 'predefined',  # Mode de filtre par défaut
        'heatmap_columns': None,  # Colonnes sélectionnées pour les heatmaps
        'data_loaded': False  # Indique si les données sont chargées
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def apply_custom_css(css):
    """Applique du CSS personnalisé à l'application"""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def create_sidebar():
    """Crée la sidebar avec toutes les options de configuration"""
    with st.sidebar:
        st.markdown("---")
        st.header("⚙️ Configuration Avancée")
        # Section de personnalisation du thème
        with st.expander("🎨 Personnalisation du Thème", expanded=False):
            app_theme = st.selectbox("Thème de l'application", ["Light", "Dark", "Custom"],
                                   help="❓ Choisissez le thème visuel de l'application")
            
            if app_theme == "Custom":
                col1, col2 = st.columns(2)
                with col1:
                    primary_color = st.color_picker("Couleur principale", "#1f77b4",
                                                  help="❓ Couleur principale de l'interface")
                    background_color = st.color_picker("Arrière-plan", "#ffffff",
                                                     help="❓ Couleur d'arrière-plan")
                with col2:
                    text_color = st.color_picker("Couleur du texte", "#000000",
                                               help="❓ Couleur du texte principal")
                    secondary_color = st.color_picker("Couleur secondaire", "#ff7f0e",
                                                    help="❓ Couleur d'accentuation")
                
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
        
        # Configuration des graphiques
        with st.expander("📊 Paramètres des Graphiques", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.graph_settings['width'] = st.slider("Largeur", 400, 1200, 800,
                                                                   help="❓ Largeur des graphiques en pixels")
                st.session_state.graph_settings['height'] = st.slider("Hauteur", 300, 900, 600,
                                                                    help="❓ Hauteur des graphiques en pixels")
            with col2:
                st.session_state.graph_settings['title_size'] = st.slider("Taille titre", 10, 24, 16,
                                                                        help="❓ Taille de police des titres")
                st.session_state.graph_settings['axis_size'] = st.slider("Taille axes", 8, 20, 14,
                                                                       help="❓ Taille de police des axes")
            
            st.session_state.graph_settings['legend_size'] = st.slider("Taille légende", 8, 20, 12,
                                                                     help="❓ Taille de police des légendes")
            st.session_state.max_graphs_per_row = st.slider("Graphiques par ligne", 1, 4, 2,
                                                          help="❓ Nombre maximum de graphiques par ligne")
            st.session_state.max_categories = st.slider("Max catégories", 5, 50, 20,
                                                      help="❓ Nombre maximum de catégories pour les variables catégorielles")
        
        # Configuration des valeurs manquantes (AMÉLIORÉE - RÉVERSIBLE)
        with st.expander("🔍 Gestion des Valeurs Manquantes", expanded=False):
            st.session_state.missing_threshold = st.slider(
                "Seuil de valeurs manquantes (%)", 0, 100, 60,
                help="❓ Pourcentage maximum de valeurs manquantes autorisé par ligne"
            )
            
            # Option pour appliquer/retenir le filtrage
            missing_action = st.radio(
                "Action sur les valeurs manquantes:",
                ["Afficher uniquement", "Exclure définitivement"],
                help="❓ 'Afficher uniquement' garde les données originales, 'Exclure définitivement' les supprime"
            )
            
            st.session_state.missing_action = missing_action
            st.info(f"Les lignes avec plus de {st.session_state.missing_threshold}% de valeurs manquantes seront {missing_action.lower()}")
        
        # Chargement des fichiers
        st.markdown("---")
        st.header("📁 Chargement des Données")
        uploaded_files = st.file_uploader("Sélectionnez les fichiers à analyser", 
                                        type=['csv', 'txt', 'xlsx', 'xls'], accept_multiple_files=True,
                                        help="❓ Chargez un ou plusieurs fichiers CSV, TXT ou Excel contenant vos données")
        
        # Traitement des fichiers uploadés (AMÉLIORÉ - CONSERVATION DES ORIGINALES)
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.uploaded_files[uploaded_file.name] = df
                    st.session_state.original_data[uploaded_file.name] = df.copy()  # Conserve les originales
                    st.session_state.filtered_data[uploaded_file.name] = df.copy()  # Copie pour filtrage
                    st.session_state.data_loaded = True
                    st.success(f"✅ {uploaded_file.name} chargé ({len(df)} lignes, {len(df.columns)} colonnes)")
        
        # Mise à jour de la liste de toutes les colonnes
        if st.session_state.uploaded_files:
            all_columns = set()
            for df in st.session_state.uploaded_files.values():
                all_columns.update(df.columns.tolist())
            st.session_state.all_columns = sorted(list(all_columns))
        
        # Sélection du fichier d'échantillonnage
        if st.session_state.uploaded_files:
            file_names = list(st.session_state.uploaded_files.keys())
            st.markdown("---")
            sampling_file = st.selectbox("📋 Fichier d'échantillonnage", file_names,
                                       help="❓ Sélectionnez le fichier contenant les données d'échantillonnage")
            
            if sampling_file:
                st.session_state.sampling_file = sampling_file
                df = st.session_state.uploaded_files[sampling_file]
                
                # Sélection de la colonne ID
                id_options = df.columns.tolist()
                st.session_state.id_column = st.selectbox("🔑 Colonne ID", id_options, index=0,
                                                        help="❓ Colonne contenant les identifiants uniques")
        
        # FILTRES DE BASE (AMÉLIORÉS - RÉVERSIBLES)
        if st.session_state.uploaded_files and st.session_state.all_columns:
            st.markdown("---")
            with st.expander("🔍 Filtres de Base", expanded=True):
                st.subheader("Filtres par Colonne")
                
                # Sélection des colonnes à filtrer
                filterable_cols = st.multiselect("Colonnes à filtrer", st.session_state.all_columns,
                                               help="❓ Sélectionnez les colonnes sur lesquelles appliquer des filtres")
                
                # Appliquer les filtres pour chaque colonne sélectionnée
                for col in filterable_cols:
                    # Trouver le type de la colonne
                    col_type = "numeric"
                    for df in st.session_state.uploaded_files.values():
                        if col in df.columns:
                            if not pd.api.types.is_numeric_dtype(df[col]):
                                col_type = "categorical"
                                break
                            # Vérifier si c'est catégoriel malgré être numérique
                            if df[col].nunique() <= st.session_state.max_categories:
                                col_type = "categorical"
                                break
                    
                    if col_type == "numeric":
                        # Trouver les valeurs min et max à travers tous les fichiers
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
                        st.session_state.filters[col] = ('range', selected_range)
                    
                    elif col_type == "categorical":
                        # Récupérer toutes les valeurs uniques à travers tous les fichiers
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
                        st.session_state.filters[col] = ('categories', selected_vals)
        
        # FILTRES AVANCÉS (AMÉLIORÉS)
        if st.session_state.uploaded_files and st.session_state.all_columns:
            with st.expander("🔧 Filtres Avancés", expanded=False):
                st.subheader("Type de Filtre Avancé")
                
                # Case à cocher pour choisir le type de filtre
                st.session_state.filter_mode = st.radio(
                    "Sélectionnez le type de filtre à utiliser:",
                    ["Prédéfini", "Personnalisé"],
                    index=0,
                    help="❓ Choisissez le type de filtre avancé à appliquer"
                )
                
                if st.session_state.filter_mode == "Prédéfini":
                    st.subheader("Filtres Prédéfinis")
                    
                    # Exemples de filtres prédéfinis avec paramètres ajustables
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
                                                 list(predefined_filters.keys()),
                                                 help="❓ Choisissez un filtre prédéfini")
                    
                    if selected_filter:
                        st.write(f"**Description:** {predefined_filters[selected_filter]['description']}")
                        st.code(predefined_filters[selected_filter]['expression'])
                        
                        # Paramètre de seuil ajustable
                        threshold_value = st.number_input("Valeur du seuil", value=0.1, step=0.1,
                                                        help="❓ Valeur du seuil pour le filtre")
                        
                        # Mapping des colonnes
                        st.subheader("Mapping des Colonnes")
                        col_mapping = {}
                        col1_mapping = st.selectbox("Colonne 1 (col1)", [""] + st.session_state.all_columns,
                                                  help="❓ Sélectionnez la première colonne")
                        col2_mapping = st.selectbox("Colonne 2 (col2)", [""] + st.session_state.all_columns,
                                                  help="❓ Sélectionnez la deuxième colonne")
                        
                        if col1_mapping:
                            col_mapping['col1'] = col1_mapping
                        if col2_mapping:
                            col_mapping['col2'] = col2_mapping
                        
                        filter_name = st.text_input("Nom du filtre", f"filtre_{selected_filter.lower()}",
                                                  help="❓ Donnez un nom à votre filtre")
                        
                        if st.button("➕ Ajouter le filtre prédéfini"):
                            if filter_name and col1_mapping and col2_mapping:
                                st.session_state.advanced_filters[filter_name] = {
                                    'expression': predefined_filters[selected_filter]['expression'],
                                    'column_mapping': col_mapping,
                                    'type': 'predefined',
                                    'threshold': threshold_value
                                }
                                st.success(f"Filtre '{filter_name}' ajouté")
                
                else:  # Mode personnalisé
                    st.subheader("Filtre Personnalisé")
                    
                    filter_name = st.text_input("Nom du filtre", "mon_filtre_personnalise",
                                              help="❓ Donnez un nom significatif à votre filtre")
                    filter_expression = st.text_area("Expression du filtre", 
                                                   "abs(col1 - col2) < 0.1",
                                                   help="❓ Utilisez col1, col2, ... comme variables génériques")
                    
                    # Mapping des colonnes
                    st.subheader("Mapping des Colonnes")
                    col_mapping = {}
                    num_columns = st.slider("Nombre de colonnes à utiliser", 1, 10, 2,
                                          help="❓ Nombre de colonnes à utiliser dans l'expression")
                    
                    for i in range(1, num_columns + 1):
                        col_mapping[f'col{i}'] = st.selectbox(
                            f"Colonne {i}", 
                            [""] + st.session_state.all_columns,
                            help=f"❓ Sélectionnez la colonne réelle pour col{i}"
                        )
                    
                    if st.button("➕ Ajouter le filtre personnalisé"):
                        if filter_name and filter_expression:
                            st.session_state.advanced_filters[filter_name] = {
                                'expression': filter_expression,
                                'column_mapping': col_mapping,
                                'type': 'custom'
                            }
                            st.success(f"Filtre '{filter_name}' ajouté")
                
                # Afficher les filtres actifs
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
        
        # NOUVELLE SECTION : Sélection des colonnes pour les heatmaps
        if st.session_state.uploaded_files and st.session_state.all_columns:
            st.markdown("---")
            with st.expander("🔥 Configuration des Heatmaps", expanded=False):
                st.subheader("Sélection des Colonnes pour les Heatmaps")
                
                # Sélection multiple des colonnes pour les heatmaps
                numeric_cols = []
                for df in st.session_state.uploaded_files.values():
                    numeric_cols.extend(df.select_dtypes(include=[np.number]).columns.tolist())
                numeric_cols = sorted(list(set(numeric_cols)))
                
                heatmap_columns = st.multiselect(
                    "Colonnes à inclure dans les heatmaps",
                    numeric_cols,
                    default=numeric_cols[:min(15, len(numeric_cols))],
                    help="❓ Sélectionnez les colonnes numériques à utiliser dans les heatmaps"
                )
                
                st.session_state.heatmap_columns = heatmap_columns
                st.info(f"{len(heatmap_columns)} colonnes sélectionnées pour les heatmaps")
        
        # Boutons de contrôle (AMÉLIORÉS)
        if st.session_state.uploaded_files:
            st.header("🎛️ Contrôles")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Réinitialiser les filtres"):
                    st.session_state.filters = {}
                    st.session_state.advanced_filters = {}
                    # Restaurer les données originales
                    for file_name in st.session_state.original_data:
                        st.session_state.filtered_data[file_name] = st.session_state.original_data[file_name].copy()
                    st.rerun()
            with col2:
                if st.button("📊 Appliquer les filtres"):
                    # Appliquer les filtres sur une copie des données originales
                    for file_name in st.session_state.original_data:
                        original_df = st.session_state.original_data[file_name]
                        filtered_df = apply_basic_filters(original_df, st.session_state.filters)
                        
                        # Appliquer les filtres avancés
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

def create_main_interface():
    """Crée l'interface principale de l'application"""
    st.title("📊 Analyse Avancée des Résultats d'Entraînement IA")
    st.markdown("""
    **Application modulaire pour l'analyse approfondie des résultats d'entraînement de modèles d'IA.**
    Chargez vos fichiers, configurez les visualisations et explorez vos données.
    """)
    
    # Vérification des données chargées
    if not st.session_state.uploaded_files:
        st.info("💡 Veuillez charger des fichiers de données dans la sidebar pour commencer.")
        return
    
    # Utiliser les données filtrées au lieu des originales
    if st.session_state.sampling_file and st.session_state.sampling_file in st.session_state.filtered_data:
        df = st.session_state.filtered_data[st.session_state.sampling_file]
    else:
        # Fallback aux données originales
        df = st.session_state.uploaded_files[list(st.session_state.uploaded_files.keys())[0]]
    
    # Calcul des métriques avancées
    initial_rows = len(st.session_state.original_data.get(st.session_state.sampling_file, df))
    high_missing_rows, missing_percentage = calculate_missing_rows(df, st.session_state.missing_threshold)
    
    # Calcul des métriques après filtrage
    filtered_rows = len(df)
    removed_rows = initial_rows - filtered_rows
    removed_percentage = (removed_rows / initial_rows) * 100 if initial_rows > 0 else 0
    
    # Affichage des métriques
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    with col1:
        st.metric("Fichiers chargés", len(st.session_state.uploaded_files), border=True, chart_type="bar")
    with col2:
        st.metric("Enregistrements totaux", initial_rows, border=True, chart_type="bar")
    with col3:
        st.metric("Colonnes disponibles", len(st.session_state.all_columns), border=True, chart_type="bar")
    with col4:
        st.metric("Filtres actifs", len(st.session_state.filters) + len(st.session_state.advanced_filters), border=True, chart_type="bar")
    with col5:
        st.metric("Lignes supprimées", f"{removed_rows} ({removed_percentage:.1f}%)", border=True, chart_type="bar")
    with col6:
        st.metric("Lignes à valeurs manquantes", f"{high_missing_rows} ({missing_percentage:.1f}%)", border=True, chart_type="bar")
    
    # Affichage des données filtrées
    with st.expander("👀 Aperçu des Données Filtrées", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"**{len(df)}** enregistrements après filtrage")
    
    # Onglets d'analyse (AMÉLIORÉS)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Distribution", "🔗 Corrélations", "📊 Catégoriel", 
        "📋 Échantillonnage", "🔥 Heatmaps", "⚡ Avancé"
    ])
    
    # Onglet Distribution
    with tab1:
        st.header("Analyse de Distribution")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()
            selected_col = st.selectbox("Colonne à analyser", all_cols, key="dist_col",
                                      help="❓ Sélectionnez la colonne à visualiser")
        with col2:
            plot_types = ["histogram", "box", "violin", "density", "bar", "pie"]
            plot_type = st.selectbox("Type de visualisation", plot_types, key="dist_type",
                                   help="❓ Choisissez le type de graphique")
        with col3:
            # Options spécifiques selon le type de graphique
            nbins = 30
            density = False
            y_col_option = None
            
            if plot_type == "histogram":
                nbins = st.slider("Nombre d'intervalles", 5, 100, 30, key="dist_nbins",
                                help="❓ Nombre de barres dans l'histogramme")
                density = st.checkbox("Afficher la densité", key="dist_density",
                                    help="❓ Superposer une courbe de densité")
            elif plot_type == "bar":
                # Pour le diagramme en barres, proposer une colonne Y optionnelle
                y_col_option = st.selectbox("Colonne Y (optionnel)", [None] + all_cols, key="dist_y_col",
                                          help="❓ Colonne pour l'axe Y (laisser vide pour un diagramme de fréquences)")
        
        # Options avancées
        col1, col2 = st.columns(2)
        with col1:
            color_col = st.selectbox("Couleur", [None] + all_cols, key="dist_color",
                                   help="❓ Colonne pour colorer les données")
        with col2:
            facet_col = st.selectbox("Facettes", [None] + all_cols, key="dist_facet",
                                   help="❓ Colonne pour créer des sous-graphiques")
        
        if selected_col:
            fig = create_custom_plot(
                df, selected_col, y_col=y_col_option, plot_type=plot_type,
                color_col=color_col, facet_col=facet_col,
                theme_settings=st.session_state.theme_settings,
                fig_size=st.session_state.graph_settings,
                nbins=nbins, density=density
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"dist_chart_{selected_col}_{plot_type}")
    
    # Onglet Corrélations
    with tab2:
        st.header("Analyse des Corrélations")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()
            x_col = st.selectbox("Variable X", all_cols, key="corr_x",
                               help="❓ Variable pour l'axe X")
        with col2:
            y_col = st.selectbox("Variable Y", all_cols, key="corr_y",
                               help="❓ Variable pour l'axe Y")
        with col3:
            corr_types = ["scatter", "line", "density", "pairplot"]
            corr_type = st.selectbox("Type de graphique", corr_types, key="corr_type",
                                   help="❓ Type de visualisation de corrélation")
        
        # Options avancées pour les corrélations
        col1, col2, col3 = st.columns(3)
        with col1:
            color_col = st.selectbox("Couleur pour corrélation", [None] + all_cols, key="corr_color",
                                   help="❓ Colonne pour colorer les points")
        with col2:
            facet_col = st.selectbox("Facettes pour corrélation", [None] + all_cols, key="corr_facet",
                                   help="❓ Colonne pour créer des sous-graphiques")
        with col3:
            if corr_type == "scatter":
                trendline_options = [None, "ols", "lowess", "expanding", "rolling"]
                trendline = st.selectbox("Ligne de tendance", trendline_options, key="corr_trendline",
                                       help="❓ Type de ligne de tendance à afficher")
            else:
                trendline = None
        
        if x_col and y_col:
            fig = create_custom_plot(
                df, x_col, y_col, plot_type=corr_type,
                color_col=color_col, facet_col=facet_col, trendline=trendline,
                theme_settings=st.session_state.theme_settings,
                fig_size=st.session_state.graph_settings
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"corr_chart_{x_col}_{y_col}_{corr_type}")
                
                # Calcul de la corrélation si possible
                if x_col in numeric_cols and y_col in numeric_cols and x_col != y_col:
                    # Corrélation de Pearson
                    pearson_corr = df[x_col].corr(df[y_col])
                    
                    # Corrélations Brice
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
        
        # Détection des colonnes catégorielles
        categorical_cols = []
        for col in df.columns:
            if df[col].nunique() <= st.session_state.max_categories:
                categorical_cols.append(col)
        
        if not categorical_cols:
            st.warning("Aucune colonne catégorielle détectée. Augmentez le 'Max catégories' dans la sidebar.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                cat_col = st.selectbox("Colonne catégorielle", categorical_cols, key="cat_col",
                                     help="❓ Colonne contenant des données catégorielles")
            with col2:
                cat_plot_types = ["bar", "pie", "box", "violin", "histogram"]
                cat_plot_type = st.selectbox("Type de graphique", cat_plot_types, key="cat_plot_type",
                                           help="❓ Type de visualisation pour données catégorielles")
            with col3:
                value_col = st.selectbox("Colonne de valeurs", [None] + numeric_cols, key="cat_value",
                                       help="❓ Colonne numérique pour l'axe Y (optionnel)")
            
            # Options avancées pour les données catégorielles
            col1, col2 = st.columns(2)
            with col1:
                color_col = st.selectbox("Couleur pour catégoriel", [None] + all_cols, key="cat_color",
                                       help="❓ Colonne pour colorer les données")
            with col2:
                facet_col = st.selectbox("Facettes pour catégoriel", [None] + all_cols, key="cat_facet",
                                       help="❓ Colonne pour créer des sous-graphiques")
            
            if cat_col:
                # Préparer les données
                if value_col and cat_plot_type in ["box", "violin"]:
                    # Graphique de distribution par catégorie
                    if cat_plot_type == "box":
                        fig = px.box(df, x=cat_col, y=value_col, color=color_col, facet_col=facet_col,
                                   title=f"{value_col} par {cat_col}")
                    else:
                        fig = px.violin(df, x=cat_col, y=value_col, color=color_col, facet_col=facet_col,
                                     box=True, title=f"{value_col} par {cat_col}")
                else:
                    # Graphique de fréquences
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
                
                # Appliquer les paramètres de thème
                fig.update_layout(
                    width=st.session_state.graph_settings['width'],
                    height=st.session_state.graph_settings['height'],
                    font_size=st.session_state.theme_settings['font_size']
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"cat_chart_{cat_col}_{cat_plot_type}")
                
                # Statistiques descriptives
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
            
            # Configuration de l'analyse
            col1, col2, col3 = st.columns(3)
            with col1:
                interval_prefix = st.text_input("Préfixe intervalles", "Interval_", key="int_prefix",
                                              help="❓ Préfixe des colonnes d'intervalles (ex: 'Interval_')")
            with col2:
                stat_prefixes = st.text_input("Préfixes stats", "Ech_count_,Ech_mean_,Ech_std_,Ech_min_,Ech_max_", 
                                            key="stat_prefix", help="❓ Préfixes des statistiques, séparés par des virgules")
            with col3:
                num_prefix = st.text_input("Préfixe nombre", "Numb_intervals_", key="num_prefix",
                                         help="❓ Préfixe de la colonne nombre d'intervalles")
            
            stat_list = [s.strip() for s in stat_prefixes.split(",") if s.strip()]
            
            if st.button("🔍 Analyser l'échantillonnage", type="primary"):
                with st.spinner("Analyse en cours..."):
                    st.session_state.analysis_results = analyze_sampling_data_improved(
                        df_sampling, interval_prefix, stat_list, num_prefix
                    )
            
            # Affichage des résultats
            if st.session_state.analysis_results:
                st.success(f"✅ {len(st.session_state.analysis_results)} suffixes analysés")
                
                # Sélection des suffixes et statistiques
                col1, col2 = st.columns(2)
                with col1:
                    available_suffixes = list(st.session_state.analysis_results.keys())
                    selected_suffixes = st.multiselect("Suffixes à visualiser", 
                                                     available_suffixes, 
                                                     default=available_suffixes[:min(2, len(available_suffixes))],
                                                     help="❓ Sélectionnez un ou plusieurs suffixes à comparer")
                with col2:
                    available_stats = set()
                    for suffix in selected_suffixes:
                        if suffix in st.session_state.analysis_results:
                            available_stats.update(st.session_state.analysis_results[suffix]['data'].keys())
                    
                    selected_stats = st.multiselect("Statistiques à visualiser", 
                                                  list(available_stats),
                                                  default=list(available_stats)[:min(2, len(available_stats))],
                                                  help="❓ Sélectionnez les statistiques à afficher")
                
                # Type de visualisation
                viz_type = st.radio("Type de visualisation", ["line", "bar"], horizontal=True,
                                  help="❓ Choisissez le type de graphique")
                
                if selected_suffixes and selected_stats:
                    fig = create_sampling_visualization(
                        st.session_state.analysis_results, selected_suffixes, selected_stats,
                        viz_type, st.session_state.theme_settings, st.session_state.graph_settings
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="sampling_chart")
                        
                        # Options d'export
                        col1, col2 = st.columns(2)
                        with col1:
                            pdf_name = st.text_input("Nom du PDF", "analyse_echantillonnage.pdf",
                                                  help="❓ Nom du fichier PDF à exporter")
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
    
    # NOUVEL ONGLET : Heatmaps
    with tab5:
        st.header("🔥 Analyse par Heatmaps")

        # Récupérer uniquement les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.warning("❌ Aucune colonne numérique trouvée dans le dataset. Les heatmaps nécessitent des colonnes numériques.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                heatmap_type = st.selectbox(
                    "Type de heatmap",
                    ["heatmap", "heatmap_brice", "heatmap_brice1"],
                    help="❓ Sélectionnez le type de matrice de corrélation à afficher"
                )

            with col2:
                # Sélection du nombre maximum de colonnes à afficher par défaut
                max_default_cols = st.slider("Nombre de colonnes par défaut", 5, 30, 15,
                                           help="❓ Nombre de colonnes à utiliser par défaut")

            # SECTION CRITIQUE : Sélection des colonnes
            st.subheader("Sélection des colonnes numériques")

            if not numeric_cols:
                st.error("Aucune colonne numérique disponible.")
            else:
                # Colonnes sélectionnées par défaut (les 15 premières colonnes numériques)
                default_selected = numeric_cols[:min(max_default_cols, len(numeric_cols))]

                # Multiselect pour choisir les colonnes
                selected_columns = st.multiselect(
                    "Colonnes à inclure dans la heatmap:",
                    options=numeric_cols,
                    default=default_selected,
                    help="❓ Sélectionnez les colonnes numériques à utiliser pour la heatmap"
                )

                st.info(f"**{len(selected_columns)}** colonnes sélectionnées sur **{len(numeric_cols)}** disponibles")

                # Afficher un aperçu des colonnes sélectionnées
                if selected_columns:
                    with st.expander("📋 Aperçu des colonnes sélectionnées"):
                        st.write("Colonnes sélectionnées:", ", ".join(selected_columns))

                        # Aperçu statistique des colonnes sélectionnées
                        st.write("**Résumé statistique:**")
                        st.dataframe(df[selected_columns].describe(), use_container_width=True)

            # Options avancées pour les heatmaps
            st.subheader("Options d'affichage")
            col1, col2, col3 = st.columns(3)
            with col1:
                show_values = st.checkbox("Afficher les valeurs", value=True,
                                        help="❓ Afficher les valeurs numériques dans les cellules")
            with col2:
                color_scale = st.selectbox("Échelle de couleurs", 
                                         ["RdBu_r", "Viridis", "Plasma", "Inferno", "Blues", "Reds"],
                                         help="❓ Choisissez l'échelle de couleurs")
            with col3:
                fig_width = st.slider("Largeur du graphique", 600, 1200, 800,
                                    help="❓ Largeur de la heatmap en pixels")

            # Bouton de génération avec gestion d'erreurs améliorée
            if st.button("🔄 Générer la heatmap", type="primary"):
                if len(selected_columns) < 2:
                    st.error("❌ Sélectionnez au moins 2 colonnes numériques pour générer une heatmap.")
                else:
                    with st.spinner("Calcul de la matrice de corrélation..."):
                        try:
                            # Vérification supplémentaire des types de données
                            valid_columns = []
                            for col in selected_columns:
                                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                                    # Vérifier qu'il n'y a pas trop de valeurs manquantes
                                    if df[col].notna().sum() >= 2:  # Au moins 2 valeurs non-nulles
                                        valid_columns.append(col)

                            if len(valid_columns) < 2:
                                st.error(f"❌ Seules {len(valid_columns)} colonnes valides. Besoin d'au moins 2 colonnes numériques avec des données.")
                            else:
                                # Création de la heatmap avec les colonnes validées
                                fig = create_custom_plot(
                                    df, None, None, plot_type=heatmap_type,
                                    theme_settings=st.session_state.theme_settings,
                                    fig_size={'width': fig_width, 'height': 600},
                                    heatmap_columns=valid_columns,
                                    max_heatmap_cols=len(valid_columns)
                                )

                                if fig:
                                    # Ajuster l'échelle de couleurs
                                    fig.update_layout(
                                        coloraxis=dict(colorscale=color_scale)
                                    )

                                    if not show_values:
                                        fig.update_traces(texttemplate='')

                                    st.plotly_chart(fig, use_container_width=True, key=f"{heatmap_type}_chart")

                                    # Informations supplémentaires
                                    st.success(f"✅ Heatmap générée avec {len(valid_columns)} colonnes numériques")

                                    # Statistiques sur les données utilisées
                                    with st.expander("📊 Statistiques des données utilisées"):
                                        st.write(f"**Colonnes utilisées:** {', '.join(valid_columns)}")
                                        st.write(f"**Nombre de lignes:** {len(df)}")
                                        st.write(f"**Valeurs manquantes:** {df[valid_columns].isnull().sum().sum()}")

                                    # Informations sur le type de heatmap
                                    if heatmap_type == "heatmap":
                                        st.info("""
                                        🔍 **Heatmap standard (Pearson)**: 
                                        - Mesure la corrélation linéaire entre variables
                                        - Valeurs entre -1 (corrélation négative) et +1 (corrélation positive)
                                        - 0 indique aucune corrélation linéaire
                                        """)
                                    elif heatmap_type == "heatmap_brice":
                                        st.info("""
                                        🔍 **Heatmap Brice**: 
                                        - Corrélation basée sur les variations premières (dérivées)
                                        - Capture les relations dans les tendances
                                        - Utile pour les séries temporelles et données séquentielles
                                        """)
                                    else:
                                        st.info("""
                                        🔍 **Heatmap Brice1**: 
                                        - Corrélation basée sur les variations centrées
                                        - Similarité dans les patterns de fluctuation
                                        - Moins sensible aux valeurs extrêmes
                                        """)

                                    # Téléchargement de la matrice de corrélation
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
                                    st.error("❌ Erreur lors de la génération de la heatmap. Vérifiez les données.")

                        except Exception as e:
                            st.error(f"❌ Erreur lors de la création de la heatmap: {str(e)}")
                            st.info("💡 **Conseils de dépannage:**")
                            st.write("- Vérifiez que toutes les colonnes sélectionnées sont numériques")
                            st.write("- Assurez-vous qu'il n'y a pas de valeurs manquantes excessives")
                            st.write("- Essayez avec moins de colonnes si le dataset est volumineux")

            # Section d'aide et d'explications
            with st.expander("❓ Aide sur les heatmaps"):
                st.markdown("""
                ### Guide d'utilisation des heatmaps

                **1. Sélection des colonnes:**
                - Seules les colonnes numériques sont disponibles
                - Sélectionnez au moins 2 colonnes pour une heatmap valide
                - Les colonnes avec trop de valeurs manquantes sont exclues

                **2. Types de heatmaps:**
                - **Heatmap standard**: Corrélation de Pearson (linéaire)
                - **Heatmap Brice**: Corrélation basée sur les variations
                - **Heatmap Brice1**: Corrélation basée sur les variations centrées

                **3. Interprétation des couleurs:**
                - 🔴 Rouge: Corrélation positive forte
                - 🔵 Bleu: Corrélation négative forte  
                - ⚪ Blanc: Peu ou pas de corrélation

                **4. Résolution des problèmes:**
                - Si erreur, réduisez le nombre de colonnes
                - Vérifiez les types de données dans l'onglet "Avancé"
                - Exportez les données pour inspection externe
                """)

    
    # Onglet Avancé
    with tab6:
        st.header("⚡ Outils d'Analyse Avancée")
        
        # Sous-onglets pour l'analyse avancée
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "📈 Statistiques", "🔍 Outliers", "⏰ Séries Temporelles", "📊 Clustering"
        ])
        
        with subtab1:
            st.subheader("📈 Analyse Statistique Complète")
            
            if st.button("📊 Générer le rapport statistique complet"):
                with st.spinner("Génération du rapport..."):
                    # Statistiques descriptives
                    st.write("### Statistiques Descriptives")
                    st.dataframe(df.describe(), use_container_width=True)
                    
                    # Informations sur les types de données
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
                    st.dataframe(pd.DataFrame(type_info), use_container_width=True)
                    
                    # Matrice de corrélation
                    st.write("### Matrice de Corrélation (Pearson)")
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 1:
                        corr_matrix = numeric_df.corr()
                        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                      color_continuous_scale='RdBu_r',
                                      title="Matrice de Corrélation entre Variables Numériques")
                        st.plotly_chart(fig, use_container_width=True, key="advanced_corr_matrix")
                    
                    # Top des corrélations
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
                        st.dataframe(corr_df.nlargest(10, 'Abs_Correlation'), use_container_width=True)
        
        with subtab2:
            st.subheader("🔍 Détection des Outliers")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Colonne à analyser", numeric_cols,
                                          help="❓ Sélectionnez une colonne numérique pour détecter les outliers")
                
                if selected_col:
                    outliers, lower_bound, upper_bound = detect_outliers_iqr(df, selected_col)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Outliers détectés", len(outliers), border=True, chart_type="bar")
                    with col2:
                        st.metric("Borne inférieure", f"{lower_bound:.2f}", border=True, chart_type="bar")
                    with col3:
                        st.metric("Borne supérieure", f"{upper_bound:.2f}", border=True, chart_type="bar")
                    
                    # Graphique des outliers
                    fig = px.box(df, y=selected_col, title=f"Distribution de {selected_col} avec outliers")
                    fig.add_hline(y=lower_bound, line_dash="dash", line_color="red", 
                                annotation_text="Borne inférieure")
                    fig.add_hline(y=upper_bound, line_dash="dash", line_color="red",
                                annotation_text="Borne supérieure")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if len(outliers) > 0:
                        st.write("### Détail des Outliers")
                        st.dataframe(outliers[[selected_col]], use_container_width=True)
            else:
                st.warning("Aucune colonne numérique disponible pour l'analyse des outliers")
        
        with subtab3:
            st.subheader("⏰ Analyse des Séries Temporelles")
            
            # Détection des colonnes de date
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
                    date_col = st.selectbox("Colonne de date", date_cols,
                                          help="❓ Sélectionnez la colonne contenant les dates")
                with col2:
                    value_col = st.selectbox("Colonne de valeurs", numeric_cols,
                                           help="❓ Sélectionnez la colonne à analyser")
                
                if date_col and value_col:
                    try:
                        df_temp = df.copy()
                        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                        fig = create_time_series_analysis(df_temp, date_col, value_col)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erreur lors de la création de la série temporelle: {e}")
            else:
                st.info("ℹ️ Aucune colonne de date détectée. Les colonnes de date doivent être au format datetime.")
        
        with subtab4:
            st.subheader("📊 Analyse par Clustering")
            
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect("Colonnes pour le clustering", numeric_cols,
                                             default=numeric_cols[:min(3, len(numeric_cols))],
                                             help="❓ Sélectionnez au moins 2 colonnes numériques")
                
                n_clusters = st.slider("Nombre de clusters", 2, 10, 3,
                                     help="❓ Nombre de groupes à créer")
                
                if len(selected_cols) >= 2 and st.button("🔍 Appliquer le clustering"):
                    with st.spinner("Clustering en cours..."):
                        fig, clustered_df = create_cluster_analysis(df, selected_cols, n_clusters)
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Statistiques par cluster
                            st.write("### Statistiques par Cluster")
                            cluster_stats = clustered_df.groupby('Cluster')[selected_cols].mean()
                            st.dataframe(cluster_stats, use_container_width=True)
            else:
                st.warning("Il faut au moins 2 colonnes numériques pour effectuer un clustering")
        
        # Section export des données
        st.markdown("---")
        st.subheader("💾 Export des Données")
        
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.radio("Format d'export", ["CSV", "Excel", "JSON"], horizontal=True,
                                   help="❓ Choisissez le format de fichier pour l'export")
            export_name = st.text_input("Nom du fichier", "donnees_analyse",
                                      help="❓ Nom de base du fichier exporté (sans extension)")
        
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

def main():
    """Fonction principale de l'application"""
    
    # Vérifier si on est en production
    IS_PRODUCTION = os.getenv('STREAMLIT_SERVER_RUNNING', 'false').lower() == 'true'
    if IS_PRODUCTION:
        configure_production_settings()
    
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
    st.markdown("**BRICE KENGNI ZANGUIM** • *Dernière mise à jour: 2025-01-23*")

if __name__ == "__main__":
    main()
