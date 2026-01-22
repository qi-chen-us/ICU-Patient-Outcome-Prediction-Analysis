"""
Utility functions to support ICU outcome analysis, including:
- Exploratory visualization
- Statistical hypothesis testing
- Threshold-based classification evaluation
- Feature importance and correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy.stats import mannwhitneyu
from scipy.stats import fisher_exact
import matplotlib.patches as mpatches

#Visualize the target distribution
def plot_target_distribution(df, target_col, title):
  if isinstance(df, pd.Series):
    df = pd.DataFrame(df)  # Convert series to df
  
  ax = sns.countplot(x=target_col, data=df, hue=target_col)
  plt.xlabel(target_col)
  plt.title(title)
  plt.ylabel('Number of Observations')
  
  total = len(df)
  for container in ax.containers:
    labels = [f'{int(v)} ({v/total:.1%})' for v in container.datavalues]
    ax.bar_label(container, labels=labels)
  plt.show()


#Function to create measured indicator and drop original columns
def create_measured_flag(X_data, cols, flag_col_name, drop_columns):
  df = X_data.copy()
  df[flag_col_name] = df[cols].notnull().any(axis=1).astype(int)
  if drop_columns == True:
    df = df.drop(columns = cols, axis = 1)
  return df


# Function to create the boxplots
def plot_boxplots(df, numeric_cols, target_col):
  num_features = len(numeric_cols)
  n_cols = 4
  n_rows = (num_features // n_cols) + (num_features % n_cols > 0)

  plt.figure(figsize = (n_cols*4, n_rows*3))
  for i, feature in enumerate(numeric_cols, 1):
      plt.subplot(n_rows, n_cols, i)
      sns.boxplot(y = df[feature].dropna(),
                  x = df[target_col],
                  hue = df[target_col],
                  showmeans = True,
                  meanprops = {'marker':'D','markeredgecolor':'red', 'markerfacecolor':'red', 'markersize':8})
      plt.legend().remove()
      plt.xticks( ticks = [0, 1], labels = ['Survived (0)', 'Died (1)'])
      plt.title(feature)
      plt.tight_layout()
  plt.show()


# Function to create KDE plots
def plot_kde_plots(df, numeric_cols, target_col):
  num_features = len(numeric_cols)
  n_cols = 4
  n_rows = (num_features // n_cols) + (num_features % n_cols > 0)

  plt.figure(figsize = (n_cols*4, n_rows*3))
  for i, feature in enumerate(numeric_cols, 1):
      plt.subplot(n_rows, n_cols, i)
      sns.kdeplot(x = df[feature].dropna(),
                  hue = df[target_col],
                  fill = True)
      #plt.legend().remove()
      plt.legend(title = 'Outcome', labels = ['Survived (0)', 'Died (1)'], bbox_to_anchor=(1.05, 1), loc='upper left')
      plt.title(feature)
      plt.tight_layout()
  plt.show()


# Function to create bar plots for binary features
def plot_binary_feature_bars(df, binary_features, target_col):
  total_patients = len(df)

  for f in binary_features:
    # Crosstab counts
    counts = pd.crosstab(df[f], df[target_col])

    ax = counts.plot(kind = 'bar', figsize =(6,4))

    ax.set_title(f'ICU Outcome by {f}')
    ax.set_ylabel('Number of Patients')
    ax.set_xlabel(f)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
    ax.legend(title = 'Outcome', 
              labels = ['Survived (0)', 'Died (1)'], 
              bbox_to_anchor = (1.05, 1), 
              loc = 'upper left')
    
    # Extend y-axis slightly (10% higher than max count)
    y_max = counts.values.max()
    ax.set_ylim(0, y_max * 1.1)

    # Annotate bars with count + overall %
    for i, row_val in enumerate(counts.index):
      for j, col_val in enumerate(counts.columns):
        count = counts.iloc[i, j]
        prop_overall = count / total_patients
        ax.text(
          x = i + j*0.25 - 0.025,  # adjust for side-by-side bars
          y = count + 1,
          s = f'{count} ({prop_overall:.1%})',
          ha = 'center',
          va = 'bottom',
          fontsize = 9)
        
    plt.tight_layout()
    plt.show()


# Mann-Whitney U Test Function
def mann_whitney_u_test(df, outcome_col, numeric_cols):
  """Perform Mann-Whitney U test for numeric features by outcome."""
    
  # Basic validation
  if df.empty:
    raise ValueError("DataFrame is empty")
  if outcome_col not in df.columns:
    raise ValueError(f"Outcome column '{outcome_col}' not found")
    
  statistic_test_list = []
  for col in numeric_cols:
    if col not in df.columns:
      print(f"Warning: Column '{col}' not found, skipping...")
      continue

    negative = df[df[outcome_col] == 0][col].dropna()
    positive = df[df[outcome_col] == 1][col].dropna()

    # Skip if insufficient data
    if len(negative) < 5 or len(positive) < 5:
      print(f"Warning: Insufficient data for '{col}', skipping...")
      continue

    # Mann-Whitney U Test
    stat, p_value = mannwhitneyu(negative, positive, alternative = 'two-sided')

    # Median
    median_neg = negative.median()
    median_pos = positive.median()

    # Rank-biserial correlation (effect size)
    n1 = len(negative)
    n2 = len(positive)
    effect_size = 1-(2  *stat)/(n1 * n2)
    effect_size_abs = abs(effect_size)

    #Output results
    statistic_test_list.append([col, stat, p_value, median_neg, median_pos, effect_size, effect_size_abs])


  if not statistic_test_list:
    raise ValueError("No Mann-Whitney U results generated")
  
  #Output results to DataFrame
  stat_df = pd.DataFrame(statistic_test_list,
                         columns = ['Feature', 'U-statistic', 'p-value', 'Median Survived', 'Median Died', 'Effect Size', 'Effect Size Abs'])
  stat_df = stat_df.set_index('Feature')
  return stat_df

# Man-Whitney U Test Effect Size Interpretation
def interpret_effect_size(effect_size):
  if effect_size < 0.1:
      return 'Negligible'
  elif effect_size < 0.3:
      return 'Small'
  elif effect_size < 0.5:
      return 'Medium'
  else:
      return 'Large' 
  

def fisher_exact_test(df, outcome_col, binary_cols):
  """Perform Fisher's Exact Test for binary features by outcome."""

  results = []

  for col in binary_cols:
    # Temporary dataframe with non-missing values
    df_tmp = df[[col, outcome_col]].dropna()
    
    # Build contingency table
    contingency_table = pd.crosstab(
      df_tmp[col].astype(int), 
      df_tmp[outcome_col].astype(int))

    # Only process if 2x2
    if contingency_table.shape != (2, 2):
      continue
    
    # Fisher's Exact Test
    odds_ratio, p_value = fisher_exact(contingency_table)
    
    results.append([col, odds_ratio, p_value])

  # Output results to DataFrame
  results_df = pd.DataFrame(results, columns = ['Feature', 'Odds Ratio', 'p-value']).set_index('Feature')
  return results_df


# Fisher's Exact Test Odds Ratio Interpretation
def interpret_odds_ratio(odds_ratio):
  if odds_ratio >1.2:
      return 'Positive Association'
  elif odds_ratio <0.8 :
      return 'Negative Association'
  else:
      return 'No Association'
  
# Threshold Distribution Plot
def plot_threshold_distribution(y_prob, y_true, model_name = None):

  plt.figure(figsize=(8,6))
  sns.histplot(y_prob[y_true == 1], 
               color='red', 
               label = 'Died', 
               kde = True, 
               stat = 'density', 
               bins = 25)
  sns.histplot(y_prob[y_true == 0], 
               color = 'blue', 
               label = 'Survived',
               kde = True, 
               stat = 'density', 
               bins = 25)
  plt.xlabel('Predicted Probability')
  plt.ylabel('Density')
  plt.xticks(np.arange(0, 1.1, 0.1))
  plt.title(f'{model_name} - Predicted Probability Distribution by Actual Outcome')
  plt.legend()
  plt.show()


# Threshold Analysis Function
def threshold_analysis(y_prob, y_true, threshold_range):
  for t in threshold_range:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0    
    print(f"Threshold {t:.3f} -> Recall: {recall:.3f}, FPR: {fpr:.3f}, Precision: {precision:.3f}")
   

# Function to evaluate predicted model performance
def evaluate_classification_model(y_prob, threshold, y_true, model_name = None):
  #Model Name with Threshold
  title_namme = model_name + f' (Threshold={threshold})\n' if model_name else f'Threshold={threshold}\n'
  
  # Prediction
  y_pred = (y_prob >= threshold).astype(int)

  # Create a single figure with 4 subplots
  fig, axes = plt.subplots(4, 1, figsize=(6,18))

 # Confusion Matrix
  cm=confusion_matrix(y_true, y_pred)

  sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', cbar = False,
              xticklabels=['Survived', 'Died'],
              yticklabels=['Survived', 'Died'],
              ax=axes[0])
  axes[0].set_ylabel('Actual')
  axes[0].set_xlabel('Predicted')
  axes[0].set_title('Confusion Matrix')

  # Classification Report
  report_dict = classification_report(y_true, y_pred, output_dict=True)
  report_text = classification_report(y_true, y_pred)
  axes[1].axis('off')  # turn off axes
  axes[1].text(.35, 0.725, 'Classification Report', fontsize=12, fontfamily='monospace', va='top')
  axes[1].text(-0.1, 0.65, report_text, fontsize=12, fontfamily='monospace', va='top')

  # ROC Curve Plot
  fpr, tpr, thresholds = roc_curve(y_true, y_prob)
  roc_auc = auc(fpr, tpr)

  axes[2].plot(fpr, tpr, color = 'blue', label = f'ROC curve (area = {roc_auc:.2f})')
  axes[2].plot([0, 1], [0, 1], color = 'red', linestyle = '--')
  axes[2].set_xlim([0.0, 1.0])
  axes[2].set_ylim([0.0, 1.05])
  axes[2].set_xlabel('False Positive Rate')
  axes[2].set_ylabel('True Positive Rate')
  axes[2].set_title('Receiver Operating Characteristic (ROC) Curve')
  axes[2].legend(loc = 'lower right')

  # Precision-Recall Curve Plot
  precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
  pr_auc = auc(recall, precision)

  axes[3].plot(recall, precision, color = 'blue', label = f'Precision-Recall curve (area = {pr_auc:.2f})')
  axes[3].set_xlim([0.0, 1.0])
  axes[3].set_ylim([0.0, 1.05])
  axes[3].set_xlabel('Recall')
  axes[3].set_ylabel('Precision')
  axes[3].set_title('Precision-Recall (PR) Curve')
  axes[3].legend(loc = 'lower left')

  
  plt.suptitle(f'{title_namme} - Model Evaluation Metrics', fontsize=12, y=0.95)
  plt.tight_layout(rect=[0, 0, 1, 0.94])
  fig.savefig(f"./results/{model_name}_Evaluation_Metrics.png", dpi=150)
  plt.show()


# Feature Importance Plot
def plot_top25_feature_importance(coef_df, model_name):
  # Top 25 Features
  coef_df_top25 = coef_df.head(25)
  coef_df_top25 = coef_df_top25.sort_values('coef')

  red_patch = mpatches.Patch(color = 'red', label = 'Positive coefficient')
  blue_patch = mpatches.Patch(color = 'blue', label = 'Negative coefficient')

  colors = ['blue' if c < 0 else 'red' for c in coef_df_top25["coef"]]

  plt.figure(figsize = (8, 6))
  plt.barh(
      coef_df_top25["feature"],
      coef_df_top25["coef"],
      color = colors)
  plt.axvline(0)
  plt.xlabel("Coefficient")
  plt.title(f"{model_name}\nTop 25 Feature Importance")
  plt.legend(handles = [red_patch, blue_patch])
  plt.tight_layout()
  plt.savefig(f'./results/{model_name}_Top25_Feature_Importance.png', dpi=150, bbox_inches='tight')
  plt.show()


def zero_coefficient_features(coef_df):
    zero_coef_features = coef_df[coef_df['coef'] == 0]['feature'].sort_values().tolist()
    return zero_coef_features


def zero_coef_features_heatmap(X_train, zero_coef_features, correlation_threshold = 0.8, top_n = 20, model_name = None):
  # Full correlation matrix
  corr_matrix = X_train[sorted(X_train.columns)].corr()

  # Select top N zero-coefficient features if specified
  if top_n is not None and len(zero_coef_features) > top_n:
    top_zero_coef_features = zero_coef_features[:top_n]
    heatmap_title = f" Top {top_n} Zero-Coefficient Features and Their Strong Correlations (>=|{correlation_threshold}|\n {model_name} "
  else:
    top_zero_coef_features = zero_coef_features
    heatmap_title = f" Zero-Coefficient Features and Their Strong Correlations (>=|{correlation_threshold}|\n {model_name} "

  # Columns where zero-coef features have |corr| >= threshold with any other feature
  mask = (
    corr_matrix.loc[top_zero_coef_features, :].gt(correlation_threshold).any() |
    corr_matrix.loc[top_zero_coef_features, :].lt(-correlation_threshold).any()
  )

  correlated_features = corr_matrix.loc[top_zero_coef_features, mask]

  # Mask weak correlations for cleaner visualization
  corr_display = correlated_features.mask(
    (correlated_features < correlation_threshold) & (correlated_features > -correlation_threshold)
  )

  # Heatmap
  plt.figure(figsize = (16,6))
  sns.heatmap(corr_display.fillna(0), 
              annot = True, 
              cmap = 'coolwarm', 
              fmt = ".1f", 
              center = 0, 
              linewidths = 0.2,
              annot_kws = {"size":10})
  plt.ylabel('Zero-Coefficient Features')
  plt.title(heatmap_title)
  plt.show()




def high_correlation_pairs_zero_coefficient_features(X_train, coef_df, correlation_threshold = 0.8):
  # Full correlation matrix
  corr_matrix = X_train[sorted(X_train.columns)].corr()


  # Find highly correlated pairs
  high_corr_pairs_idx = np.where(np.abs(corr_matrix) > correlation_threshold)

  # Filter out duplicates and self-correlation
  high_corr_pairs = [
      (corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
      for x, y in zip(*high_corr_pairs_idx) if x != y and x < y]

  # Convert to DataFrame
  high_corr_df = pd.DataFrame(high_corr_pairs, columns=["Feature_1", "Feature_2", "Correlation"])
  high_corr_df = high_corr_df.sort_values(by="Correlation", ascending=False).reset_index(drop=True)


  # Merge high correlation pairs with logistic regression coefficients
  high_corr_df = high_corr_df.merge(coef_df.rename(columns = {'feature': 'Feature_1', 'coef': 'Coef_1', 'abs_coef': 'Abs_Coef_1 '}), on = 'Feature_1')
  high_corr_df = high_corr_df.merge(coef_df.rename(columns = {'feature': 'Feature_2', 'coef': 'Coef_2', 'abs_coef': 'Abs_Coef_2 '}), on ='Feature_2')
  high_corr_df = high_corr_df.sort_values(by='Correlation', ascending=False)
  high_corr_df = high_corr_df[['Feature_1','Feature_2','Correlation', 'Coef_1',   'Coef_2' ]].sort_values(['Feature_1','Feature_2'])

  high_corr_df_zero_coef = high_corr_df[(high_corr_df.Coef_1==0) | (high_corr_df.Coef_2==0)]
  return high_corr_df_zero_coef


def zero_coef_remove_features(high_corr_df_zero_coef):
  features_to_remove = []

  for _, row in high_corr_df_zero_coef.iterrows():
    f1 = row['Feature_1'].lower()
    f2 = row['Feature_2'].lower()

    c1 = row['Coef_1']
    c2 = row['Coef_2']

    # Case 1: only one has zero coefficient
    if c1 == 0 and c2 != 0:
      features_to_remove.append(row['Feature_1'])
    elif c2 == 0 and c1 != 0:
      features_to_remove.append(row['Feature_2'])
    
    # Case 2: both zero
    elif c1 == 0 and c2 == 0:
      # Keep derived features: ratio, delta, range
      if any(x in f1 for x in ['ratio','delta','range']):
        features_to_remove.append(row['Feature_2'])
      elif any(x in f2 for x in ['ratio','delta','range']):
        features_to_remove.append(row['Feature_1'])
      # Otherwise follow priority: last > highest > lowest
      elif 'last' in f1:
        features_to_remove.append(row['Feature_2'])
      elif 'last' in f2:
        features_to_remove.append(row['Feature_1'])
      elif 'highest' in f1:
       features_to_remove.append(row['Feature_2'])
      elif 'highest' in f2:
       features_to_remove.append(row['Feature_1'])
      elif 'lowest' in f1:
       features_to_remove.append(row['Feature_2'])
      elif 'lowest' in f2:
        features_to_remove.append(row['Feature_1'])
      else:
        # If nothing matches, just drop f2
        features_to_remove.append(row['Feature_2'])
  return sorted(list(set(features_to_remove)))
