# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 20:06:06 2025

@author: fupf
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 08:42:05 2025

@author: fupf
"""
import os
import numpy as np

# import seaborn as sns
import matplotlib.pyplot as plt
# import seaborn as sns
# from osgeo import gdal
import pandas as pd
# from datetime import datetime,timedelta
import xgboost as xgb
from math import sqrt
# from sklearn.cluster import KMeans
# import shappry
import shap
import warnings
from matplotlib.ticker import FormatStrFormatter
# import scipy.signal
# import matplotlib.colors as mcolors
from scipy.interpolate import make_interp_spline
# LightGBM 模型训练（正确使用早停）
# from lightgbm import early_stopping, log_evaluation
warnings.filterwarnings('ignore')
#%%

data_path=r'G:\Data\降水和土壤水分数据提取\2021年1-12月耕地降水和delta_sm的数据\数据提取\全国数据\2017-2022\模型模拟数据\0~7cm\土壤分类\数据\0~7cm全国9大农业分区40个地形质地分类数据.csv'
out_path=r'G:\Data\降水和土壤水分数据提取\2021年1-12月耕地降水和delta_sm的数据\数据提取\全国数据\2017-2022\模型模拟数据\0~7cm\土壤分类\训练结果\【新】40个土壤分类中挑选15个主要类型贡献结果'
out_path1=r'G:\Data\降水和土壤水分数据提取\2021年1-12月耕地降水和delta_sm的数据\数据提取\全国数据\2017-2022\模型模拟数据\0~7cm\土壤分类\训练结果\V2_40个土壤分类中挑选15个主要类型贡献结果'
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


all_data_reclass1['TM'] = pd.to_datetime(all_data_reclass1['TM'], format='%Y_%m_%d', errors='coerce')
all_data_reclass1['month'] = all_data_reclass1['TM'].dt.month





# joint_class='Silty Clay Loam_1'
# 遍历每个分类
for joint_class in data_info_tiqu:
    print(f"  处理分类: {joint_class}")
    
    joint_class='Loam_1'
    
    # 构建完整输出路径
    full_dir = os.path.join(out_path1, joint_class)
    
    # 检查文件夹是否已存在
    # if os.path.exists(full_dir):
    #     print(f"文件夹 '{full_dir}' 已存在，跳过处理。")
    #     continue  # 跳过本次循环，不处理该分类
    
    # 文件夹不存在，创建它
    os.makedirs(full_dir)
    print(f"文件夹 '{full_dir}' 已创建。")
    # 存储所有最优参数和评估结果
    all_best_params_df = []
    
    
    class_data = all_data_reclass1[all_data_reclass1['joint_class'] == joint_class]
    
    # 获取所有 fenqu_ID
    fenqu_ids = sorted(class_data['fenqu_ID'].unique())
    
    # 外层循环：遍历每个分区
    for fenqu_id in fenqu_ids:
        print(f"\n=== 处理分区 {fenqu_id} ===")
        fenqu_id=1.0
        
        # 提取当前分区数据
        fenqu_data = class_data[class_data['fenqu_ID'] == int(fenqu_id)]
        
        # 如果样本太少，跳过（避免交叉验证失败）
        if len(fenqu_data) < threshold:
            print("样本数不足，跳过...")
            continue
        #shap计算样本数据
        sample_df = fenqu_data.sample(frac=0.2, random_state=42, replace=False)
    
    
        full_df_path=out_path+'\\{}'.format(joint_class)+'\\分区{}土壤分类{}_测试集SHAP值.csv'.format(fenqu_id,joint_class)
        
        full_df=pd.read_csv(full_df_path)
        
        # =====================================
        # 🔍 使用 SHAP 进行贡献分析
        # =====================================
        columns = [
                      'TP','ASML1', 
                      'STL1','TE', 
                      'EVI', 'NDVI','LAI',
                       'SP','T2M', 
                       'WU10M', 'WV10M',
                      'BD','POROSITY'
                      ]
        
        shap_columns = [f"shap_{col}" for col in columns]
        full_df_tiqu = full_df[shap_columns]
        full_df_tiqu_numpy = full_df[shap_columns].to_numpy()
        
        shap_values = full_df_tiqu_numpy
    
        result = full_df_tiqu.abs().mean()
    
    
    
        import matplotlib as mpl

        mpl.rcParams['font.family'] = 'Times New Roman'
        # mpl.rcParams['mathtext.fontset'] = 'custom'
        mpl.rcParams['mathtext.rm'] = 'Times New Roman'
        mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
        mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    
        plt.rcParams["axes.unicode_minus"] = False  # 防止负号显示异常
        # 全局设置 Times New Roman 字体
        plt.rcParams['text.usetex'] = False  # 确保不使用 LaTeX（除非你有需求）
        
        
        # 1. 特征重要性条形图（全局解释）#################################
        fig=plt.figure(figsize=(8, 8))
        # 直接绘制条形图，不接收返回值（因为它可能是 None）
        shap.summary_plot(shap_values, sample_df[columns], plot_type="bar", show=False)
        
        # 使用 gca() 获取当前坐标轴
        ax = plt.gca()
        
        # === 关闭网格线 ===
        ax.grid(False)
        
        # === 设置四周边框颜色和粗细 ===
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')      # 更明确的颜色设置
            ax.spines[spine].set_linewidth(2)        # 设置线宽为1
        
        # === 设置标签和字体加粗 ===
        # ax.set_xlabel("Mean |SHAP Value|", fontsize=26, fontweight='bold')
        # ax.set_ylabel("Features", fontsize=26, fontweight='bold')
        
        ax.set_xlabel("")   # 清空 x 轴标题
        ax.set_ylabel("")   # 清空 y 轴标题
        
        ax.text(0.95, 0.05, joint_class, 
                transform=ax.transAxes,
                fontsize=32,
                fontweight='bold',
                fontname='Times New Roman',
                verticalalignment='bottom',
                horizontalalignment='right',
                color='black')
        
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # === 设置刻度标签大小和加粗 ===
        ax.tick_params(axis='both', which='major', labelsize=20, width=1, color='black')
        ax.tick_params(axis='y', which='major', labelsize=20, width=1, color='black')
        
        # 重新设置 y 轴标签以支持 fontweight
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        ax.set_yticklabels(y_labels, fontsize=30, fontweight='bold')
        
        # 可选：x 轴标签也加粗
        x_labels = [label.get_text() for label in ax.get_xticklabels()]
        ax.set_xticklabels(x_labels, fontsize=30, fontweight='bold',rotation=45)
        ax.xaxis.set_label_coords(0.5, -0.15)  # 调整 -0.15 到你想要的确切位置
        
        # === 调整布局并显示 ===
        plt.tight_layout()
        # plt.show()
        plt.savefig(out_path1+'\\{}'.format(joint_class)+"\\分区{}土壤分类{}_delta_sm_bar.png".format(fenqu_id,joint_class), dpi=300, bbox_inches='tight')
        plt.close()       
        
        
    
    
        pre_name='TP'
        soil_name='ASML1'
        st_name='STL1'
        
        # 假定 shap_values 是一个二维数组，每个特征对应一行。
        sorted_idx = result.argsort()[::-1]  # 根据重要性对特征进行降序排列
        
        # 提取最重要的两个特征
        top_features = [columns[i] for i in sorted_idx[:5]]
        
        # pre_name=top_features[0]
        # soil_name=top_features[1]
        
        
        
        
        
        # 4.贡献最大5个要素tipping######################################
        
        # -----------------------------
        # 数据预处理：按 'ERA5_precipitation' 分组统计
        # -----------------------------
        def plot_shap_scatter_by_value_col(
            full_df,
            value_col,
            fenqu_id,
            joint_class,
            out_path,
            bin_size=10,
            threshold=30
        ):
            """
            根据指定的 value_col 对数据进行分组、筛选显著区间，并绘制 SHAP 值散点平滑图，
            保存为 PNG 文件，文件名和横轴标签根据 value_col 动态调整。
        
            参数:
                full_df (pd.DataFrame): 包含 'shap_{value_col}' 和 {value_col} 列的 DataFrame
                value_col (str): 要分析的变量列名（如 'TP', 'STL2' 等）
                fenqu_id (str or int): 分区 ID
                joint_class (str): 土壤分类名称
                out_path (str): 输出目录路径
                bin_size (int): 初始分组 bin 大小，默认 20
                threshold (int): 显著组最小样本数阈值，默认 30
            """
            
            # -----------------------------
            # Step 1: 定义横坐标标签映射
            # -----------------------------
            xlabel_map = {
                'TP': 'Precipitation(mm)',
                'STL1': 'Soil Temperature(K)',
                'TE': 'Evaporation(mm)',
                'ASML1': r"Soil Moisture (m$^{3}$·m$^{-3}$)",
                'T2M': '2m Temperature(K)',
                'WU10M':r'10m_u_component_of_wind(m·s$^{-1}$)',
                'WV10M':r'10m_v_component_of_wind(m·s$^{-1}$)',
                'SP':'Surface Pressure(Pa)',
                'LAI':r'Leaf Area Index(m$^{2}$·s$^{-2}$)',
                'BD':r'Bulk Density(g·cm$^{-3}$)',
                'POROSITY':'Porosity(%)',
                'EVI':'Enhanced Vegetation Index',
                'NDVI':'Normalized Difference Vegetation Index'
            }
        
        
            xlabel = xlabel_map.get(value_col, f'{value_col}')  # 默认用列名
        
            # -----------------------------
            # Step 2: 检查必要列是否存在
            # -----------------------------
            if value_col not in full_df.columns:
                raise ValueError(f"列 '{value_col}' 不存在于 DataFrame 中")
            shap_col = f'shap_{value_col}'
            if shap_col not in full_df.columns:
                raise ValueError(f"列 '{shap_col}' 不存在于 DataFrame 中")
        
        
            # -----------------------------
            # Step 3: 筛选显著分组
            # -----------------------------
            def filter_df_by_significant_groups(df, col, bin_size, threshold):
                
                # max_val = full_df[value_col].max()
                
                max_val = df[col].max()
                upper_bound = (int(max_val) // bin_size + 1) * bin_size
                bins = list(range(0, upper_bound + 1, bin_size))
                labels = [f'{i}-{i+bin_size}' for i in bins[:-1]]
                bin_labels = pd.cut(
                    df[col],
                    bins=bins,
                    labels=labels,
                    right=False,
                    include_lowest=True
                )
                group_counts = bin_labels.value_counts().sort_index()
                significant_groups = group_counts[group_counts > threshold].index
                filtered_df = df[bin_labels.isin(significant_groups)].copy()
                filtered_df['group'] = bin_labels[bin_labels.isin(significant_groups)]
                return filtered_df, significant_groups
        
            filtered_df, significant_groups = filter_df_by_significant_groups(full_df, value_col, bin_size, threshold)
        
            if filtered_df.empty:
                print(f"警告：{value_col} 在分区 {fenqu_id}, 类别 {joint_class} 下无显著分组，跳过绘图。")
                return
        
            # -----------------------------
            # Step 4: 进一步细分为 bins 并计算统计量
            # -----------------------------
            num_bins = len(significant_groups) * 10
            filtered_df['bin'] = pd.cut(filtered_df[value_col], bins=num_bins, labels=False)
            bin_centers = filtered_df.groupby('bin')[value_col].mean()
            bin_mean = filtered_df.groupby('bin')[shap_col].mean()
            bin_upper = filtered_df.groupby('bin')[shap_col].quantile(0.9)
            bin_lower = filtered_df.groupby('bin')[shap_col].quantile(0.1)
        
            bin_upper = bin_upper.reindex(bin_centers.index)
            bin_lower = bin_lower.reindex(bin_centers.index)
        
        
            # -----------------------------
            # Step 6: 平滑曲线
            # -----------------------------
            X_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 300)
            spl_mean = make_interp_spline(bin_centers, bin_mean, k=3)
            spl_upper = make_interp_spline(bin_centers, bin_upper, k=3)
            spl_lower = make_interp_spline(bin_centers, bin_lower, k=3)
            
            Y_mean_smooth = spl_mean(X_smooth)
            Y_upper_smooth = spl_upper(X_smooth)
            Y_lower_smooth = spl_lower(X_smooth)
            
            # 找出所有 Y_mean_smooth 穿过 0 的位置（包括从正到负、负到正）
            def find_zero_crossings(x, y):
                """
                找到 y=0 的所有交点（通过符号变化 + 线性插值）
                返回：交点 x 坐标列表
                """
                # 移除 NaN（虽然平滑后一般没有，但保险起见）
                valid = ~np.isnan(y)
                x = x[valid]
                y = y[valid]
                
                # 找出符号变化的位置（y[i] * y[i+1] < 0）
                sign_changes = np.where(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
                
                crossings = []
                for i in sign_changes:
                    # 线性插值估计零点
                    x1, x2 = x[i], x[i+1]
                    y1, y2 = y[i], y[i+1]
                    # 避免除零（理论上不会，因符号不同）
                    if y2 != y1:
                        x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
                        crossings.append(x_zero)
                
                # 可选：检查是否有 y 精确等于 0 的点（罕见，但可加）
                exact_zeros = x[np.isclose(y, 0, atol=1e-12)]
                crossings.extend(exact_zeros)
                
                # 去重并排序
                crossings = sorted(set(crossings))
                return crossings
            
            # 调用函数
            cross_x_list = find_zero_crossings(X_smooth, Y_mean_smooth)
            
            # -----------------------------
            # Step 7: 绘图
            # -----------------------------
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 1. 灰色填充：上下界之间（不确定性带）
            ax.fill_between(X_smooth, Y_lower_smooth, Y_upper_smooth, color='gray', alpha=0.2)
            
            # 2. 绘制平滑曲线
            ax.plot(X_smooth, Y_mean_smooth, color='blue', linewidth=2, label='SHAP mean value')
            ax.plot(X_smooth, Y_upper_smooth, color='gray', linewidth=1)
            ax.plot(X_smooth, Y_lower_smooth, color='gray', linewidth=1)
            
            for idx, cross_x in enumerate(cross_x_list):
                
                # 3. 红色垂直线：cross_x
                ax.axvline(x=cross_x, color='red', linestyle='--', linewidth=1.5)
                
                # 5. 标记交点
                ax.plot(cross_x, 0, 'ro', markersize=10, label='Tipping point')
            
                # 动态计算文本偏移量，使得标签不互相遮盖
                # 可以根据实际需要调整下面的数字
                x_offset = (-40 if idx % 2 == 0 else -80)  # 左右交替移动
                y_offset = (50 if idx % 2 == 0 else -50)  # 按照索引增加y方向的偏移，防止上下重叠
                
                # 6. 添加注释（数值标签）
                ax.annotate(f'{cross_x:.2f}',
                            xy=(cross_x, 0),
                            xytext=(x_offset, y_offset),  # 动态计算偏移
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5", color='k', lw=2),
                            fontsize=30, color='k', fontweight='bold')
            
            # 4. 横轴零线
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
            
            # 7. 分区域填充：左侧负贡献（粉红），右侧正贡献（蓝色）
            # --- 左侧：bin_centers < cross_x 且 bin_mean < 0 → 填充到 y=0
            left_mask = (Y_mean_smooth < 0)
            if np.any(left_mask):
                ax.fill_between(X_smooth[left_mask], Y_mean_smooth[left_mask], 0,
                                facecolor='pink', alpha=0.6, label='Negative Region', interpolate=True)
            
            # --- 右侧：bin_centers > cross_x 且 bin_mean > 0 → 填充到 y=0
            right_mask = (Y_mean_smooth > 0)
            if np.any(right_mask):
                ax.fill_between(X_smooth[right_mask], Y_mean_smooth[right_mask], 0,
                                facecolor='lightblue', alpha=0.6, label='Positive Region', interpolate=True)
            
            # 8. 美化
            ax.grid(False)
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color('black')
                ax.spines[spine].set_linewidth(2)
            
            # ax.set_xlabel(xlabel, fontsize=30, fontweight='bold')
            # ax.set_ylabel('SHAP Value', fontsize=30, fontweight='bold')
            
            ax.set_xlabel("")   # 清空 x 轴标题
            ax.set_ylabel("")   # 清空 y 轴标题
            
            ax.text(0.95, 0.05, joint_class, 
                    transform=ax.transAxes,
                    fontsize=32,
                    fontweight='bold',
                    fontname='Times New Roman',
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    color='black')
            
            # 获取当前所有句柄和标签
            handles, labels = ax.get_legend_handles_labels()
            
            # 使用 dict 保持顺序并去重（Python 3.7+ 字典有序）
            # by_label = dict(zip(labels, handles))
            
            # 图例不显示
            ax.legend().set_visible(False)
            
            # 重新设置图例
            # ax.legend(by_label.values(), by_label.keys(), fontsize=20, ncol=4, bbox_to_anchor=(0.5, -0.1), loc='upper center')
            # ax.legend(by_label.values(), by_label.keys(),fontsize=20)
            # ax.legend(fontsize=20)
            plt.xticks(fontsize=30, fontweight='bold')
            plt.yticks(fontsize=30, fontweight='bold')
            plt.tight_layout()
        
            # -----------------------------
            # Step 8: 保存图片
            # -----------------------------
            save_dir = os.path.join(out_path1, str(joint_class))
            os.makedirs(save_dir, exist_ok=True)
            filename = "分区{}土壤分类{}_{}_delta_sm_scatter.png".format(fenqu_id, joint_class, value_col)
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            plt.close()
        
            print("✅ 已保存 ")
        
        
        for para in top_features:
        
            plot_shap_scatter_by_value_col(
                full_df,
                para,
                fenqu_id,
                joint_class,
                out_path,
                bin_size=10,
                threshold=30
            )
        
        
        
        
        # 6.第二个依赖图，使用 'ERA5_start_soil' 作为交互变量 ################
        plt.figure(figsize=(8, 8))  # 再次调整图表大小
        shap.dependence_plot(soil_name, shap_values, sample_df[columns], interaction_index=st_name, show=False)
        
        # 使用 gca() 获取当前坐标轴
        ax = plt.gca()
        
        # === 关闭网格线 ===
        ax.grid(False)
        
        # === 设置四周边框颜色和粗细 ===
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')      # 更明确的颜色设置
            ax.spines[spine].set_linewidth(2)        # 设置线宽为1
        
        # 设置标题和坐标轴标签
        # plt.title("SHAP Dependence Plot for ERA5_precipitation with Interaction of ERA5_start_soil", fontsize=14)
        # plt.xlabel('Precipitation(mm)', fontsize=18, fontweight='bold')
        # plt.ylabel("SHAP Value", fontsize=18, fontweight='bold')
        
        ax.set_xlabel("")   # 清空 x 轴标题
        ax.set_ylabel("")   # 清空 y 轴标题
        # 设置坐标轴刻度和网格线
        plt.xticks(fontsize=30, fontweight='bold',rotation=45)
        plt.yticks(fontsize=30, fontweight='bold')
        # plt.grid(True, linestyle='--', alpha=0.7)
        ax.text(0.95, 0.05, joint_class, 
                transform=ax.transAxes,
                fontsize=32,
                fontweight='bold',
                fontname='Times New Roman',
                verticalalignment='bottom',
                horizontalalignment='right',
                color='black')
        # === 关键：获取 colorbar 并设置字体 ===
       
        # 依赖图通常会添加一个 colorbar，它是 fig.axes 的最后一个
        if len(plt.gcf().axes) > 1:
            cbar_ax = plt.gcf().axes[-1]
            
            # === 关键：设置 colorbar 刻度值保留 2 位小数 ===
            cbar_ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            cbar_ax.tick_params(labelsize=16)
            for label in cbar_ax.get_yticklabels():
                label.set_fontsize(30)
                label.set_fontweight('bold')
            # 设置 colorbar 标签
            if cbar_ax.get_ylabel():
                cbar_ax.set_ylabel(cbar_ax.get_ylabel(), fontsize=30, fontweight='bold')
                # cbar_ax.yaxis.set_label_coords(2.3, 0.5)  # 可选：调整 colorbar 标题位置
        # 调整布局以避免标签被截断
        plt.tight_layout()
        
        plt.savefig(out_path1+'\\{}'.format(joint_class)+"\\分区{}土壤分类{}_delta_sm_two_dependence.png".format(fenqu_id,joint_class), dpi=300, bbox_inches='tight')
        plt.close()
        # 显示图像
        # plt.show()
      
