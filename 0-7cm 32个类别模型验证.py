# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:38:40 2025

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
from matplotlib.ticker import FuncFormatter  # 导入用于格式化的工具
import statsmodels.api as sm
import warnings
import optuna
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
#%%

data_path=r'G:\Data\降水和土壤水分数据提取\2021年1-12月耕地降水和delta_sm的数据\数据提取\全国数据\2017-2022\模型模拟数据\0~7cm\土壤分类\数据\0~7cm全国9大农业分区40个地形质地分类数据.csv'
out_path=r'G:\Data\降水和土壤水分数据提取\2021年1-12月耕地降水和delta_sm的数据\数据提取\全国数据\2017-2022\模型模拟数据\0~7cm\土壤分类\训练结果\V2_40个土壤分类中挑选15个主要类型贡献结果'
data_info_path=r'G:\Data\降水和土壤水分数据提取\2021年1-12月耕地降水和delta_sm的数据\数据提取\全国数据\2017-2022\模型模拟数据\0~7cm\土壤分类\数据\0~7cm不同农业分区下地形质地分类统计结果.xlsx'

data_info=pd.read_excel(data_info_path)
data_info_tiqu=data_info['土壤分类'].tolist()[0:15] #98.17%
threshold=data_info.iloc[14,1]

all_data_reclass1=pd.read_csv(data_path)

all_data_reclass1['evaporation']=all_data_reclass1['evaporation']*1000
all_data_reclass1['ERA5_precipitation']=all_data_reclass1['ERA5_precipitation']*1000
all_data_reclass1=all_data_reclass1[(all_data_reclass1['ERA5_start_soil']>=0) 
                                    ]


all_data_reclass1=all_data_reclass1.rename(
    columns={'ERA5_precipitation':'TP',#total precipitation/m
             'ERA5_start_soil':'ASML1',#antecedent soil moisture/m3m-3
             'ERA5_delta_soil':'DSML1',#delta_soil_moisture/m3m-3
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
              'TP','ASML1', 'DSML1',
              'STL1','TE', 
              'EVI', 'NDVI','LAI',
               'SP','T2M', 'WU10M', 'WV10M',
              'BD','POROSITY'
              ]

def root_mean_squared_error(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False  # 防止负号显示异常


for joint_class in data_info_tiqu:
    print(f"  处理分类: {joint_class}")
    
    joint_class='Loam_1'
    
    class_data = all_data_reclass1[all_data_reclass1['joint_class'] == joint_class]
    
    # 获取所有 fenqu_ID
    fenqu_ids = sorted(class_data['fenqu_ID'].unique())
    
    # 外层循环：遍历每个分区
    for fenqu_id in fenqu_ids:
        print(f"\n=== 处理分区 {fenqu_id} ===")
        fenqu_id=1.0
        
        # 提取当前分区数据
        fenqu_data = class_data[class_data['fenqu_ID'] == int(fenqu_id)]
        
        # kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # 如果样本太少，跳过（避免交叉验证失败）
        if len(fenqu_data) < threshold:
            print("样本数不足，跳过...")
            continue
    
        sample_df = fenqu_data.sample(frac=0.2, random_state=42, replace=False)
        
        train_log_path=out_path+'\\{}'.format(joint_class)+"\\分区{}土壤分类{}_optuna_training_log.csv".format(fenqu_id,joint_class)
        
        # 检查训练日志文件是否存在
        if not os.path.exists(train_log_path):
            print(f"警告: 文件不存在，跳过该分区 {fenqu_id}.")
            continue  # 跳过此次循环的剩余部分
            
        params_df=pd.read_csv(train_log_path)
        
        best_tril=params_df['trial_number'].iloc[0]
        # print(params_df['n_estimators'].iloc[0])
        best_params = {
            'n_estimators': params_df['n_estimators'].iloc[0],
            'max_depth': params_df['max_depth'].iloc[0],
            'learning_rate':params_df['learning_rate'].iloc[0],
            'subsample': params_df['subsample'].iloc[0],
            'colsample_bytree': params_df['colsample_bytree'].iloc[0],
            'min_child_weight': params_df['min_child_weight'].iloc[0],
            'reg_alpha': params_df['reg_alpha'].iloc[0],
            'reg_lambda': params_df['reg_lambda'].iloc[0],
        }
        final_model_xgb = xgb.XGBRegressor(**best_params)
        
        columns = [
                      'TP','ASML1', 
                      'STL1','TE', 
                      'EVI', 'NDVI','LAI',
                       'SP','T2M', 
                       'WU10M', 'WV10M',
                      'BD','POROSITY'
                      ]
        
        model = final_model_xgb.fit(sample_df[columns], sample_df['DSML1'].values)
        
        # 用于存储每折的评估结果
        # cv_results = {
        #     'fold': [],
        #     'rmse': [],
        #     'mae': [],
        #     'r2': []
        # }
        
        # X_scaled=fenqu_data[columns].to_numpy()
        # y_scaled=fenqu_data['DSML1'].values
        # # 使用最优参数重新训练，并记录每折指标
        # for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        #     X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
        #     y_train_fold, y_test_fold = y_scaled[train_idx], y_scaled[test_idx]
    
        #     final_model = xgb.XGBRegressor(**best_params, random_state=42, eval_metric='rmse')
        #     final_model.fit(X_train_fold, y_train_fold, verbose=False)
    
        #     y_pred = final_model.predict(X_test_fold)
    
        #     rmse = root_mean_squared_error(y_test_fold, y_pred)
        #     mae = mean_absolute_error(y_test_fold, y_pred)
        #     r2 = r2_score(y_test_fold, y_pred)
            
    
        #     cv_results['fold'].append(fold)
        #     cv_results['rmse'].append(rmse)
        #     cv_results['mae'].append(mae)
        #     cv_results['r2'].append(r2)
    
        # # 计算平均和标准差
        # mean_rmse = np.mean(cv_results['rmse'])
        # std_rmse = np.std(cv_results['rmse'])
        # mean_mae = np.mean(cv_results['mae'])
        # std_mae = np.std(cv_results['mae'])
        # mean_r2 = np.mean(cv_results['r2'])
        # std_r2 = np.std(cv_results['r2'])
        
        
        y_pred = model.predict(sample_df[columns])
        y_true = sample_df['DSML1'].values
        
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}')
       
        # 创建图形
        fig, ax = plt.subplots(figsize=(6, 6))
        
        
        # === 设置四周边框颜色和粗细 ===
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')      # 更明确的颜色设置
            ax.spines[spine].set_linewidth(2)        # 设置线宽为1
            
        # 创建 KDE 模型来估计密度
        kde = gaussian_kde([y_true, y_pred])
        density = kde([y_true, y_pred])
        
        
        # === 添加回归线和 95% 置信区间 ===
        X = sm.add_constant(y_true)  # 添加截距项
        ci_model = sm.OLS(y_pred, X).fit()
        
        # 生成用于绘图的平滑 x 值
        x_fit = np.linspace(y_true.min(), y_true.max(), 100)
        X_fit = sm.add_constant(x_fit)
        y_fit = ci_model.predict(X_fit)
        
        # 计算预测值的标准误（用于置信区间）
        pred_summary = ci_model.get_prediction(X_fit)
        # ci = pred_summary.conf_int(alpha=0.5)  # 95% CI
        
        # 绘制回归线和置信区间
        ax.plot(x_fit, y_fit, color='#E26E67', linewidth=3, label='Regression Line')
        # ax.fill_between(x_fit, ci[:, 0], ci[:, 1], color='gray', alpha=0.7, label='95% CI')
        
        # 绘制散点图，颜色基于密度
        scatter = ax.scatter(y_true, y_pred, c=density, cmap='coolwarm', s=20, edgecolors='none')
        
        # 添加 x=y 参考线（黑色虚线）
        x_min, x_max = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([x_min, x_max], [x_min, x_max], 'k--', linewidth=2.5,label='1:1 Line')
        
        # 格式化函数，用于保留两位小数
        def two_decimal_formatter(x, pos):
            return '%.2f' % x
        
        # 应用到x轴上
        ax.xaxis.set_major_formatter(FuncFormatter(two_decimal_formatter))
        
        # 设置标签（英文）并加粗字体和增大字体
        ax.set_xlabel('True Values', fontsize=20, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=20, fontweight='bold')
        # ax.set_title('XGBoost Model', fontsize=18, fontweight='bold')
        
        # === 设置刻度标签大小和加粗 ===
        ax.tick_params(axis='both', which='major', labelsize=18, width=1, color='black')
        ax.tick_params(axis='y', which='major', labelsize=18, width=1, color='black')
        
        # 重新设置 y 轴标签以支持 fontweight
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        ax.set_yticklabels(y_labels, fontsize=18, fontweight='bold')
        
        # 可选：x 轴标签也加粗
        x_labels = [label.get_text() for label in ax.get_xticklabels()]
        ax.set_xticklabels(x_labels, fontsize=18, fontweight='bold')
        # 添加网格线
        ax.grid(True, alpha=0.3)
        
        # 关闭图例
        # ax.legend().set_visible(False)
        
        # === 新增：创建图例，放在图像外下方，2列 ===
        legend = ax.legend(
            loc='upper center',           # 图例锚点位置（相对于 bbox_to_anchor）
            bbox_to_anchor=(0.5, -0.15),  # (x, y)：x=0.5 居中，y=-0.15 在图外下方
            ncol=2,                       # 2 列
            fontsize=16,
            frameon=False,                 # 显示边框（可选）
            fancybox=False,
            shadow=False,
            columnspacing=1.5,            # 列间距
            handletextpad=0.5             # 图标与文字间距
        )
        
        # 设置图例字体为 Times New Roman 并加粗
        for text in legend.get_texts():
            text.set_fontname('Times New Roman')
            text.set_fontweight('bold')
        # 显示测试集评价指标在左上角，使用黑色字体
        test_metrics_text = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
        ax.text(0.05, 0.95, test_metrics_text, transform=ax.transAxes, fontsize=20, fontweight='bold',
                verticalalignment='top', horizontalalignment='left', color='black')
        
        # 在右下角添加自定义文本
        ax.text(0.95, 0.05, joint_class, 
                transform=ax.transAxes,
                fontsize=20,
                fontweight='bold',
                fontname='Times New Roman',
                verticalalignment='bottom',
                horizontalalignment='right',
                color='black')
        
        # 添加颜色条（可选）
        cbar = plt.colorbar(scatter)
        # 设置 colorbar 刻度标签字体为 Times New Roman 并加粗
        cbar.set_label('Density', rotation=270, labelpad=20, fontsize=20, fontweight='bold')
        
        cbar.ax.tick_params(labelsize=18)  # 可选：统一字号
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')
        # 调整布局
        plt.tight_layout()
        plt.savefig(out_path+"\\模型验证.png",dpi=600,bbox_inches='tight')

        # plt.savefig(out_path+'\\{}'.format(joint_class)+"\\分区{}土壤分类{}第{}次模型验证1.png".format(fenqu_id,joint_class,best_tril),dpi=600,bbox_inches='tight')
        plt.close()
        print("保存成功!")
        

#%%



