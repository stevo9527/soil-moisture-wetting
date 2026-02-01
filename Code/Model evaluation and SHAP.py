# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 10:07:58 2025

@author: fupf
"""

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from scipy.stats import gaussian_kde
import os
from sklearn.model_selection import KFold
from matplotlib.ticker import FuncFormatter  # Import formatting tools
import statsmodels.api as sm
import warnings
import shap
import optuna
import matplotlib as mpl
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import train_test_split
from scipy.interpolate import make_interp_spline
#%%


data_info=pd.read_excel(data_info_path)
data_info_tiqu=data_info['土壤分类'].tolist()[0:15] #98.17%
threshold=data_info.iloc[14,1]

all_data_reclass1=pd.read_csv(data_path)

all_data_reclass1['evaporation']=all_data_reclass1['evaporation']*1000
all_data_reclass1['ERA5_precipitation']=all_data_reclass1['ERA5_precipitation']*1000
all_data_reclass1=all_data_reclass1[(all_data_reclass1['ERA5_start_soil']>=0) ]



all_data_reclass1=all_data_reclass1[(all_data_reclass1['ASML1_RSM']>0) &
                                    (all_data_reclass1['ASML1_RSM']<130) 
                                    ]

all_data_reclass1=all_data_reclass1[(all_data_reclass1['END_RSM']<=120) 
                                    ]

all_data_reclass1=all_data_reclass1[all_data_reclass1['DSML1_RSM']>1]



all_data_reclass1=all_data_reclass1.rename(
    columns={'ERA5_precipitation':'TP',#total precipitation/m
             'ASML1_RSM':'ASML1',#antecedent soil moisture/m3m-3
             'DSML1_RSM':'DSML1',#delta_soil_moisture/m3m-3
             'soil_temperature':'STL1',#Soil temperature level 1/K
             'evaporation':'TE',#total Evaporation/mm of water equivalent
             'EVI-day':'EVI',
             'NDVI-day':'NDVI',
             'leaf_area_index_low_vegetation':'LAI',#leaf_area_index_low_vegetation/m2m-2
             'pressure':'SP',#surface_pressure/Pa
             'temperature':'T2M',#2m temperature/K
             'wind_u':'WU10M',#10m_u_component_of_wind/ms-1
             'wind_v':'WV10M',#10m_V_component_of_wind/ms-1
             
             })

parameters = [
              'TP','ASML1', 
              'STL1','TE', 
              'EVI', 'NDVI','LAI',
               'SP','T2M', 'WU10M', 'WV10M',
              'BD','POROSITY'
              ]

def root_mean_squared_error(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False  # Prevent abnormal display of minus signs


joint_class='Loam_1'

full_dir = os.path.join(out_path1, joint_class)
os.makedirs(full_dir)
print(f"Folder '{full_dir}' has been created.")

class_data = all_data_reclass1[all_data_reclass1['joint_class'] == joint_class]

sample_df = class_data.sample(frac=0.1, random_state=42, replace=False)

#%%


X = class_data[parameters].values
y = class_data['DSML1'].values  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# Use MinMaxScaler instead of StandardScaler
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        'eval_metric': 'rmse',
        'verbosity': 1,
        'tree_method': 'gpu_hist'  # Enable GPU
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)

    pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, pred)
    
    return rmse  # Return average RMSE for optimization

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(xgb_objective, n_trials=50)




# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        'eval_metric': 'rmse',
        'verbosity': 1,
        'tree_method': 'gpu_hist'  # Enable GPU
    }

    fold_scores = []
    for train_idx, val_idx in kf.split(X):
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
        # y_train_fold, y_val_fold = y_scaled[train_idx], y_scaled[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train_fold, y_train_fold, verbose=False)

        pred = model.predict(X_val_fold)
        rmse = root_mean_squared_error(y_val_fold, pred)
        fold_scores.append(rmse)

    return np.mean(fold_scores)  # Return average RMSE for optimization

# # Use Optuna for optimization
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=10)

# Extract intermediate values for each trial
all_trials_data = []

for trial in study.trials:
    all_trials_data.append({
        'trial_number': trial.number,
        # 'step': step,
        'rmse': trial.value,
        **trial.params  # Save hyperparameters for this trial
    })

# Convert to DataFrame
intermediate_df = pd.DataFrame(all_trials_data)
# print(intermediate_df.head())
intermediate_df.sort_values(by='rmse', ascending=True, inplace=True)
# Save to CSV

best_tril=intermediate_df['trial_number'].iloc[0]
# print(params_df['n_estimators'].iloc[0])
best_params = {
    'n_estimators': intermediate_df['n_estimators'].iloc[0],
    'max_depth': intermediate_df['max_depth'].iloc[0],
    'learning_rate':intermediate_df['learning_rate'].iloc[0],
    'subsample': intermediate_df['subsample'].iloc[0],
    'colsample_bytree': intermediate_df['colsample_bytree'].iloc[0],
    'min_child_weight': intermediate_df['min_child_weight'].iloc[0],
    'reg_alpha': intermediate_df['reg_alpha'].iloc[0],
    'reg_lambda': intermediate_df['reg_lambda'].iloc[0],
}
final_model = xgb.XGBRegressor(**best_params)



best_params = study.best_params

# # print(f"  Best parameters: {best_params}")
final_model = xgb.XGBRegressor(**best_params)


test_cv_results = {
    'fold': [],
    'rmse': [],
    'mae': [],
    'r2': []
}

train_cv_results = {
    'fold': [],
    'rmse': [],
    'mae': [],
    'r2': []
}
# Retrain with optimal parameters and record metrics for each fold
# for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    


# # print(f"  CV10 Results - RMSE: {mean_rmse:.4f}±{std_rmse:.4f}, "
# #       f"MAE: {mean_mae:.4f}±{std_mae:.4f}, R²: {mean_r2:.4f}±{std_r2:.4f}")

# Save results


final_model.fit(X_train, y_train, verbose=False)

y_pred = final_model.predict(sample_df[parameters])
y_true = sample_df['DSML1'].values

rmse = root_mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}')



#%%

# Create KDE model to estimate density
kde = gaussian_kde([y_true, y_pred])
density = kde([y_true, y_pred])


# === Add regression line and 95% confidence interval ===
# X = sm.add_constant(y_true)  # Add intercept term
ci_model = sm.OLS(y_pred, X).fit()

# Generate smooth x values for plotting
x_fit = np.linspace(y_true.min(), y_true.max(), 100)
X_fit = sm.add_constant(x_fit)
y_fit = ci_model.predict(X_fit)

# Calculate standard error of predicted values (for confidence interval)
pred_summary = ci_model.get_prediction(X_fit)
ci = pred_summary.conf_int(alpha=0.5)  # 95% CI

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))


# === Set border color and linewidth for all four sides ===
for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color('black')      # More explicit color setting
    ax.spines[spine].set_linewidth(2)        # Set line width to 1
    
# Draw regression line and confidence interval
ax.plot(x_fit, y_fit, color='#E26E67', linewidth=3, label='Regression Line')
ax.fill_between(x_fit, ci[:, 0], ci[:, 1], color='gray', alpha=0.7, label='95% CI')

# Draw scatter plot with colors based on density
scatter = ax.scatter(y_true, y_pred, c=density, cmap='coolwarm', s=20, edgecolors='none')

# Add x=y reference line (black dashed line)
x_min, x_max = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
ax.plot([x_min, x_max], [x_min, x_max], 'k--', linewidth=2.5,label='1:1 Line')

# Formatting function to keep two decimal places
def two_decimal_formatter(x, pos):
    return '%.2f' % x

# Apply to x-axis
ax.xaxis.set_major_formatter(FuncFormatter(two_decimal_formatter))

# Set labels (English) and bold font and increase font size
ax.set_xlabel('True Values', fontsize=20, fontweight='bold')
ax.set_ylabel('Predicted Values', fontsize=20, fontweight='bold')
ax.set_title('XGBoost Model', fontsize=18, fontweight='bold')

# === Set tick label size and bold ===
ax.tick_params(axis='both', which='major', labelsize=18, width=1, color='black')
ax.tick_params(axis='y', which='major', labelsize=18, width=1, color='black')

# Reset y-axis labels to support fontweight
y_labels = [label.get_text() for label in ax.get_yticklabels()]
ax.set_yticklabels(y_labels, fontsize=18, fontweight='bold')

# Optional: also bold x-axis labels
x_labels = [label.get_text() for label in ax.get_xticklabels()]
ax.set_xticklabels(x_labels, fontsize=18, fontweight='bold')
# Add grid lines
ax.grid(True, alpha=0.3)

# Turn off legend
ax.legend().set_visible(False)

# === New: Create legend, place outside bottom of image, 2 columns ===
legend = ax.legend(
    loc='upper center',           # Legend anchor position (relative to bbox_to_anchor)
    bbox_to_anchor=(0.5, -0.15),  # (x, y): x=0.5 centered, y=-0.15 outside bottom
    ncol=2,                       # 2 columns
    fontsize=16,
    frameon=False,                 # Show border (optional)
    fancybox=False,
    shadow=False,
    columnspacing=1.5,            # Column spacing
    handletextpad=0.5             # Icon and text spacing
)

# Set legend font to Times New Roman and bold
for text in legend.get_texts():
    text.set_fontname('Times New Roman')
    text.set_fontweight('bold')
# Display test set evaluation metrics in upper left corner, using black font
test_metrics_text = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
ax.text(0.05, 0.95, test_metrics_text, transform=ax.transAxes, fontsize=20, fontweight='bold',
        verticalalignment='top', horizontalalignment='left', color='black')

# Add custom text in lower right corner
ax.text(0.95, 0.05, joint_class, 
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        fontname='Times New Roman',
        verticalalignment='bottom',
        horizontalalignment='right',
        color='black')

# Add colorbar (optional)
cbar = plt.colorbar(scatter)
# Set colorbar tick label font to Times New Roman and bold
cbar.set_label('Density', rotation=270, labelpad=20, fontsize=20, fontweight='bold')

cbar.ax.tick_params(labelsize=18)  # Optional: unified font size
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontweight('bold')
# Adjust layout
plt.tight_layout()
plt.savefig(out_path+"\\模型验证.png",dpi=600,bbox_inches='tight')

plt.savefig(full_dir+"\\50次相对湿度土壤分类{}模型验证.png".format(joint_class),dpi=1200,bbox_inches='tight')
plt.close()
print("Saved successfully!")

#%%
# =====================================
# 🔍 Use SHAP for contribution analysis
# =====================================
columns = [
              'TP','ASML1', 
              'STL1','TE', 
              'EVI', 'NDVI','LAI',
               'SP','T2M', 
               'WU10M', 'WV10M',
              'BD','POROSITY'
              ]


model = final_model.fit(sample_df[columns], sample_df['DSML1'].values)


# Create explainer object
explainer = shap.Explainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(sample_df[columns])


shap_df = pd.DataFrame(shap_values, columns=columns)

# shap result export
result = shap_df.abs().mean()
result.to_csv(full_dir+'\\50次相对湿度土壤分类{}_测试集SHAP绝对均值.csv'.format(joint_class),encoding='utf_8_sig', index=True, header=True)


# If you need to include original input feature values, you can do this:
full_df = sample_df.copy().reset_index(drop=True)
for col in columns:
    full_df[f"shap_{col}"] = shap_df[col]

# Get DataFrame of SHAP values (each column is SHAP value for each feature)
full_df.to_csv(full_dir+'\\50次相对湿度土壤分类{}_测试集SHAP值.csv'.format(joint_class),encoding='utf_8_sig',index=False)


#%%

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'


plt.rcParams["axes.unicode_minus"] = False  # Prevent abnormal display of minus signs
# Global Times New Roman font setting
plt.rcParams['text.usetex'] = False  # Ensure not using LaTeX (unless you have needs)


# 1. Feature importance bar plot (global explanation)#################################
fig=plt.figure(figsize=(8, 8))
# Directly draw bar plot, do not receive return value (because it may be None)
shap.summary_plot(shap_values, sample_df[columns], plot_type="bar", show=False)

# Use gca() to get current axes
ax = plt.gca()

# === Turn off grid lines ===
ax.grid(False)

# === Set border color and linewidth for all four sides ===
for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color('black')      # More explicit color setting
    ax.spines[spine].set_linewidth(2)        # Set line width to 1

# === Set labels and font bold ===
ax.set_xlabel("Mean |SHAP Value|", fontsize=26, fontweight='bold')
ax.set_ylabel("Features", fontsize=26, fontweight='bold')

ax.set_xlabel("")   # Clear x-axis title
ax.set_ylabel("")   # Clear y-axis title

ax.text(0.95, 0.05, joint_class, 
        transform=ax.transAxes,
        fontsize=32,
        fontweight='bold',
        fontname='Times New Roman',
        verticalalignment='bottom',
        horizontalalignment='right',
        color='black')

ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# === Set tick label size and bold ===
ax.tick_params(axis='both', which='major', labelsize=20, width=1, color='black')
ax.tick_params(axis='y', which='major', labelsize=20, width=1, color='black')

# Reset y-axis labels to support fontweight
y_labels = [label.get_text() for label in ax.get_yticklabels()]
ax.set_yticklabels(y_labels, fontsize=30, fontweight='bold')

# Optional: also bold x-axis labels
x_labels = [label.get_text() for label in ax.get_xticklabels()]
ax.set_xticklabels(x_labels, fontsize=30, fontweight='bold',rotation=45)
ax.xaxis.set_label_coords(0.5, -0.15)  # Adjust -0.15 to the exact position you want

# === Adjust layout and display ===
plt.tight_layout()
plt.show()
plt.savefig(full_dir+"\\50次相对湿度土壤分类{}_delta_sm_bar.png".format(joint_class), dpi=1200, bbox_inches='tight')
plt.close()       





