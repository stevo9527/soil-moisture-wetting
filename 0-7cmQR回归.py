# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 17:58:41 2025

@author: fupf
"""

    
#%%
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# import sns
import os
# ---------------------------------------------------------
# 1. 数据准备 (基于您提供的 JSON 数据)
# ---------------------------------------------------------

data_path=r'G:\Data\降水和土壤水分数据提取\2021年1-12月耕地降水和delta_sm的数据\数据提取\全国数据\2017-2022\模型模拟数据\0~7cm\土壤分类\数据\0~7cm全国9大农业分区40个地形质地分类和田持数据.csv'
out_path=r'G:\Data\降水和土壤水分数据提取\2021年1-12月耕地降水和delta_sm的数据\数据提取\全国数据\2017-2022\模型模拟数据\0~7cm\土壤分类\训练结果\【新】40个土壤分类中挑选15个主要类型贡献结果'
out_path1=r'G:\Data\降水和土壤水分数据提取\2021年1-12月耕地降水和delta_sm的数据\数据提取\全国数据\2017-2022\模型模拟数据\0~7cm\土壤分类\训练结果\QR回归'

merged=pd.read_csv(data_path)

merged['evaporation']=merged['evaporation']*1000
merged['ERA5_precipitation']=merged['ERA5_precipitation']*1000
merged=merged[(merged['ERA5_start_soil']>=0) ]


merged=merged.rename(
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
              'TP','ASML1_RSM', 
              'STL1','TE', 
              'EVI', 'NDVI','LAI',
               'SP','T2M', 'WU10M', 'WV10M',
              'BD','POROSITY'
              ]

joint_class='Clay Loam_1'

full_dir = os.path.join(out_path1, joint_class)
# os.makedirs(full_dir)
# print(f"文件夹 '{full_dir}' 已创建。")

class_data = merged[merged['joint_class'] == joint_class]


#%%

out_path=r'G:\Data\降水和土壤水分数据提取\2021年1-12月耕地降水和delta_sm的数据\数据提取\全国数据\2017-2022\模型模拟数据\0~7cm\土壤分类\训练结果\QR回归\v2'

joint_class='Loam_1'

full_dir = os.path.join(out_path, joint_class)
# os.makedirs(full_dir)
# print(f"文件夹 '{full_dir}' 已创建。")

class_data = merged[merged['joint_class'] == joint_class]

fenqu_ids = sorted(class_data['fenqu_ID'].unique())
fenqu_ids = [x for x in fenqu_ids if x != 5]

sector_map = {1: "A", 2: "B", 3: "C", 4: "D", 7: "F", 6: "G", 9: "H", 8: "I"}


for fenqu_id in fenqu_ids:
    print(f"\n=== 处理分区 {fenqu_id} ===")
    # fenqu_id=1.0

    # 提取当前分区数据
    fenqu_data = class_data[class_data['fenqu_ID'] == int(fenqu_id)]
    
    fenqu_data_tiqu=fenqu_data[['fenqu_ID','TP','ASML1_RSM','DSML1_RSM']]
    fenqu_data_tiqu.insert(1,'depth','0-7cm')
    
    fenqu_data_tiqu.columns = ['zone', 'depth', 'TP', 'RSM', 'delta_sm']
    
    df=fenqu_data_tiqu.copy()
    
    
    # 2. 配置参数
    config = {
        'delta_sm_threshold': 1,   # 提高一点阈值以便测试
        'tau': 0.5,                   # 使用中位数回归更稳定
        # 'rsm_bins': np.arange(0, 1.05, 0.1),
        'min_samples_per_bin': 30,
        'min_rainfall': 1.0
    }
    
    
    
    # 1. 配置参数
    delta_sm_threshold = config.get('delta_sm_threshold', 1)
    tau = config.get('tau', 0.5)
    min_samples = config.get('min_samples_per_bin', 30)
    min_rainfall = config.get('min_rainfall', 1.0)
    
    # 2. 数据预处理
    df_clean = df.copy()
    
    # 筛选有效湿化事件
    df_valid = df_clean[
        (df_clean['TP'] >= min_rainfall) & 
        (df_clean['delta_sm'] > 0)
    ].copy()
    
    
    # 步骤 1: 只保留 RSM < 120 的数据（可选：也建议 RSM >= 0）
    df_valid = df_valid[(df_valid['RSM'] >= 0) & (df_valid['RSM'] < 120)].copy()
    
    # 步骤 2: 定义自定义 bins 和 labels
    bins = [0, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    labels = [
        '0-30%',
        '30-40%',
        '40-50%',
        '50-60%',
        '60-70%',
        '70-80%',
        '80-90%',
        '90-100%',
        '100-110%',
        '110-120%'
    ]
    
    # 步骤 3: 使用 pd.cut 分箱
    df_valid['RSM_bin'] = pd.cut(
        df_valid['RSM'],
        bins=bins,
        labels=labels,
        include_lowest=True  # 确保 0 被包含在第一个 bin
    )
    
    # 可选：删除未落入任何 bin 的行（理论上不会出现，因已过滤 RSM<120 且 >=0）
    df_valid = df_valid.dropna(subset=['RSM_bin']).reset_index(drop=True)
    
    min_bin=df_valid['RSM_bin'].min()   
    max_bin=df_valid['RSM_bin'].max()
    
    min_bin_split=min_bin.split("-")[0]
    max_bin_split=max_bin.split("-")[1]
    
    
    
    # ---------------------------------------------------------
    # 2. 分位数回归计算 TP_threshold
    # ---------------------------------------------------------
   
    
    target_delta_sm_list=[1,2,3,4,5,6,7,8,9,10]
    # target_delta_sm = 10  # 有效湿化临界值 1%
    
    for target_delta_sm in target_delta_sm_list:
        
        # print(target_delta_sm)
        results_data = []
        bins = df_valid['RSM_bin'].unique()
        tau = 0.5             # 分位数
        
        all_qr_result=[]
        # print(">>> 分位数回归 (QR) 计算结果:")
        # print(f"{'Bin':<10} | {'Mean RSM':<10} | {'Intercept':<10} | {'Slope':<10} | {'TP_th (mm)':<10}")
        # print("-" * 65)
        
        for bin_label in bins:
            sub_df = df_valid[df_valid['RSM_bin'] == bin_label]
            
            col_index = sub_df.columns.get_loc('RSM_bin')
            RSM_bin = sub_df.iat[0, col_index]
            
            # QR 模型: delta_sm ~ TP
            mod = smf.quantreg('delta_sm ~ TP', sub_df)
            res = mod.fit(q=tau)
            
            beta_0 = res.params['Intercept']
            beta_1 = res.params['TP']
            
            # 计算阈值: TP_th = (1 - beta_0) / beta_1
            tp_th = (target_delta_sm - beta_0) / beta_1
            
            if 0 < tp_th < 100: # 计算RSM区间中点 
                mean_rsm = sub_df['RSM'].mean()
                
                results_data.append([mean_rsm, tp_th,RSM_bin])
                
                qr_results={
                    "Bin":bin_label,
                    "Mean RSM":round(mean_rsm,2),
                    "Intercept":round(beta_0,4),
                    "Slope":round(beta_1,4),
                    "TP_th (mm)":round(tp_th,4)
                    
                    }
            
                all_qr_result.append(qr_results)
            
            
            # print(f"{bin_label:<10} | {mean_rsm:<10.2f} | {beta_0:<10.4f} | {beta_1:<10.4f} | {tp_th:<10.4f}")
        
        
        qr_result_df=pd.DataFrame(all_qr_result)
        qr_result_df.to_csv(full_dir+'\\分区{}土壤分类{}湿度梯度增加{}QR回归结果.csv'.format(fenqu_id,joint_class,target_delta_sm),encoding='utf_8_sig', index=False, header=True)
        
        
        # 转换为 DataFrame 用于拟合
        df_fit = pd.DataFrame(results_data, columns=['x', 'y','RSM_bin'])
        # 按 RSM 从小到大排序
        df_fit = df_fit.sort_values(by='x')
        
        # --- 注意 ---
        # 提供的示例数据只有2个Bin点。2个点拟合2参数模型 R2 必然为 1，且无法计算P值（自由度不足）。
        # 为了演示完整的代码逻辑，这里手动添加一个模拟点 (假设 RSM=45, 趋势下降)
        # **实际使用时请删除下面这行代码，使用您的完整数据集**
        if len(df_fit) < 3:
            print("\n[警告] 数据点不足3个，添加模拟点仅供演示代码功能...")
            df_fit.loc[len(df_fit)] = [45.0, 0.35] 
        
        X = df_fit['x'].values
        Y = df_fit['y'].values
        
        # ---------------------------------------------------------
        # 3. 曲线拟合与统计检验 (线性化方法)
        # ---------------------------------------------------------
        
        # A. 指数衰减模型: Y = a * e^(b * X)  =>  ln(Y) = ln(a) + b * X
        #    令 Y' = ln(Y), A = ln(a), B = b
        df_fit['ln_y'] = np.log(df_fit['y'])
        exp_mod = smf.ols('ln_y ~ x', data=df_fit).fit()
        
        exp_a = np.exp(exp_mod.params['Intercept'])
        exp_b = exp_mod.params['x']
        exp_r2 = exp_mod.rsquared
        exp_p_val = exp_mod.pvalues['x']  # 斜率的显著性
        
        # print(f"\n>>> 指数模型拟合结果 (Exponential):")
        # print(f"Eq: TP_th = {exp_a:.4f} * e^({exp_b:.4f} * RSM)")
        # print(f"R2: {exp_r2:.4f}, P-value: {exp_p_val:.4e}")
        
        # B. 幂律模型: Y = a * X^b  =>  ln(Y) = ln(a) + b * ln(X)
        #    令 Y' = ln(Y), X' = ln(X), A = ln(a), B = b
        df_fit['ln_x'] = np.log(df_fit['x'])
        pow_mod = smf.ols('ln_y ~ ln_x', data=df_fit).fit()
        
        pow_a = np.exp(pow_mod.params['Intercept'])
        pow_b = pow_mod.params['ln_x']
        pow_r2 = pow_mod.rsquared
        pow_p_val = pow_mod.pvalues['ln_x']
        
        # print(f"\n>>> 幂律模型拟合结果 (Power Law):")
        # print(f"Eq: TP_th = {pow_a:.4f} * RSM ^ ({pow_b:.4f})")
        # print(f"R2: {pow_r2:.4f}, P-value: {pow_p_val:.4e}")
        
        # ---------------------------------------------------------
        # 2. 多项式回归模型 (Quadratic Model)
        # 模型公式: Y = beta_0 + beta_1 * X + beta_2 * X^2
        # ---------------------------------------------------------
        # 使用 statsmodels 的公式接口，I(RSM**2) 表示 RSM 的平方项
        poly_mod = smf.ols(formula='y ~ x + I(x**2)', data=df_fit).fit()
        
        # 提取参数
        beta_0 = poly_mod.params['Intercept']     # 截距 c
        beta_1 = poly_mod.params['x']           # 一次项系数 b
        beta_2 = poly_mod.params['I(x ** 2)']   # 二次项系数 a
        r_squared = poly_mod.rsquared             # R2
        p_values = poly_mod.pvalues               # P值
        
        # 输出统计结果
        # print(">>> 多项式模型 (二次) 拟合结果:")
        # print(f"方程: y = {beta_2:.6f} * x^2 + {beta_1:.6f} * x + {beta_0:.6f}")
        # print(f"R-squared (R2): {r_squared:.4f}")
        # print(f"P-values:\n{p_values}")
        
        muti_model_result={
            "Exp_Eq":f"TP_th = {exp_a:.4f} * e^({exp_b:.4f} * RSM)",
            "Exp_R2":f"{pow_r2:.4f}",
            "Exp_P-value":f"{pow_p_val:.4e}",
            "Pow_Eq":f"TP_th = {pow_a:.4f} * RSM ^ ({pow_b:.4f})",
            "Pow_R2":f"{pow_r2:.4f}",
            "Pow_P-value":f"{pow_p_val:.4e}",
            "Pol_Eq":f"y = {beta_2:.6f} * x^2 + {beta_1:.6f} * x + {beta_0:.6f}",
            "Pol_R2":f"{r_squared:.4f}",
            "Pol_P-value":f"{p_values[2]:.4f}",
            }
        
        muti_result_df=pd.DataFrame([muti_model_result])
        muti_result_df.to_csv(full_dir+'\\分区{}土壤分类{}梯度湿度增加{}多模型拟合结果.csv'.format(fenqu_id,joint_class,target_delta_sm),encoding='utf_8_sig', index=False, header=True)
    
        
        # ---------------------------------------------------------
        # 4. 绘图代码
        # ---------------------------------------------------------
        
        import matplotlib as mpl
        
        mpl.rcParams['font.family'] = 'Times New Roman'
        # mpl.rcParams['mathtext.fontset'] = 'custom'
        mpl.rcParams['mathtext.rm'] = 'Times New Roman'
        mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
        mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
        
        
        plt.rcParams["axes.unicode_minus"] = False  # 防止负号显示异常
        # 全局设置 Times New Roman 字体
        plt.rcParams['text.usetex'] = False  # 确保不使用 LaTeX（除非你有需求）
        
        
        # plt.figure(figsize=(10, 6), dpi=100)
        
        # 设置画布风格，类似你上传的图片（白色背景，无网格或淡网格）
        # sns.set(style="ticks") 
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        
        
        
        # B. 绘制散点 (蓝色实心点，带边框)
        ax.scatter(X, Y,  color='black', s=80, edgecolors='black', zorder=5, label='Calculated Thresholds')
        
        
        # 1. 绘制散点 (实测值)
        # plt.scatter(X, Y, color='black', marker='o', s=80, label='Calculated Thresholds (QR, τ=0.25)', zorder=5)
        
        # 2. 生成平滑曲线数据
        x_smooth = np.linspace(X.min() * 0.95, X.max() * 1.05, 100)
        
        # 计算拟合值
        y_exp_smooth = exp_a * np.exp(exp_b * x_smooth)
        y_pow_smooth = pow_a * np.power(x_smooth, pow_b)
        
        # 代入二次方程计算 y
        y_pol_smooth = beta_2 * (x_smooth**2) + beta_1 * x_smooth + beta_0
        
        
        # 3. 绘制拟合曲线
        # plt.plot(x_smooth, y_exp_smooth, 'g-', linewidth=2, label='Exponential Decay Fit')
        # plt.plot(x_smooth, y_pow_smooth, 'b--', linewidth=2, label='Power Law Fit')
        # plt.plot(x_smooth, y_pol_smooth, color='#e74c3c', linewidth=2, label='Polynomial Fit')
        
        # A. 绘制拟合曲线 (红色虚线，模仿你的第一张图)
        ax.plot(x_smooth, y_exp_smooth, 'g-', linewidth=2, label=f"Exponential Model: $R^2 = {exp_r2:.3f}, P = {exp_p_val:.3f}$")
        ax.plot(x_smooth, y_pow_smooth, 'b--', linewidth=2, label=f"Power Model: $R^2 = {pow_r2:.3f}, P = {pow_p_val:.3f}$")
        ax.plot(x_smooth, y_pol_smooth, color='#e74c3c', linewidth=2, label=f"Polynomial Model: $R^2 = {r_squared:.3f}, P = {p_values[2]:.3f}$")
        
        
        # =========================================================
        # [关键步骤] C. 添加文本标签
        # =========================================================
        # 遍历 DataFrame 的每一行
        for index, row in df_fit.iterrows():
            # label_text = f"RSM {row['RSM_bin']} ({row['y']:.2f} mm)"
            label_text = f"{row['y']:.2f}"
            x_pos = row['x']
            y_pos = row['y']
            
            # 根据 index 是偶数还是奇数，决定位置
            if index % 2 == 0:
                # 右上：x 向右，y 向上；va='bottom' 表示文字底部在 (x,y) 上方 → 视觉在点上方
                dx, dy = 1.2, -0.03
                va = 'bottom'
            else:
                # 右下：x 向右，y 向下；va='top' 表示文字顶部在 (x,y) 下方 → 视觉在点下方
                dx, dy = 1.2, +0.03
                va = 'top'
            
            ax.text(
                x_pos + dx, 
                y_pos + dy, 
                label_text,
                color='black',
                fontsize=10,
                fontweight='normal',
                ha='left',      # 文字从 x_pos+dx 开始向右写（左侧对齐）
                va=va           # 垂直对齐方式：bottom（上）或 top（下）
            )
            
            
        # 4. 图表装饰
        plt.title('Precipitation threshold at a {}% increase in soil relative humidity gradient'.format(target_delta_sm), fontsize=14)
        plt.xlabel('Soil Relative Humidity (RSM, %)', fontsize=12)
        plt.ylabel('Precipitation Threshold ($TP_{th}$, mm)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 5. 添加统计信息文本框
        # stats_text = (
        #     f"Exponential Model:\n"
        #     f"$y = {exp_a:.2f} e^{{{exp_b:.3f}x}}$\n"
        #     f"$R^2 = {exp_r2:.3f}, P = {exp_p_val:.3f}$\n\n"
        #     f"Power Law Model:\n"
        #     f"$y = {pow_a:.2f} x^{{{pow_b:.3f}}}$\n"
        #     f"$R^2 = {pow_r2:.3f}, P = {pow_p_val:.3f}$\n\n"
        #     f"Polynomial Model:\n"
        #     f"$y = {beta_2:.1e}x^2 + {beta_1:.3f}x {beta_0:.3f}$\n" # 使用科学计数法显示极小的二次项
        #     f"$R^2 = {r_squared:.3f}, P = {p_values[2]:.3f}$"
           
        # )
        
        # # 放置文本框 (坐标根据数据范围自动调整，或者固定位置)
        # box_props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        # plt.text(0.75, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
        #          verticalalignment='top', bbox=box_props)
        
        zone_label = sector_map.get(fenqu_id, str(fenqu_id))  # 若不在字典中，回退为字符串数字
        
        ax.text(0.9, 0.05, f'{joint_class} at zone {zone_label}',
                transform=ax.transAxes,
                fontsize=32,
                fontweight='bold',
                fontname='Times New Roman',
                verticalalignment='bottom',
                horizontalalignment='right',
                color='black')
        
        plt.legend( fontsize=10)
        plt.tight_layout()
        
        plt.savefig(full_dir+"\\分区{}土壤分类{}_{}-{}湿度梯度增加{}降水阈值.png".format(fenqu_id,joint_class,min_bin_split,max_bin_split,target_delta_sm), dpi=300, bbox_inches='tight')
        plt.close() 
        # 显示图形
        # plt.show()       



#%%

class_data_tiqu=class_data[['fenqu_ID','TP','ASML1_RSM','DSML1_RSM']]
class_data_tiqu.insert(1,'depth','0-7cm')

class_data_tiqu.columns = ['zone', 'depth', 'TP', 'RSM', 'delta_sm']

df=class_data_tiqu.copy()


# 2. 配置参数
config = {
    'delta_sm_threshold': 1,   # 提高一点阈值以便测试
    'tau': 0.50,                   # 使用中位数回归更稳定
    # 'rsm_bins': np.arange(0, 1.05, 0.1),
    'min_samples_per_bin': 30,
    'min_rainfall': 1.0
}



# 1. 配置参数
delta_sm_threshold = config.get('delta_sm_threshold', 1)
tau = config.get('tau', 0.50)
min_samples = config.get('min_samples_per_bin', 30)
min_rainfall = config.get('min_rainfall', 1.0)

# 2. 数据预处理
df_clean = df.copy()

# 筛选有效湿化事件
df_valid = df_clean[
    (df_clean['TP'] >= min_rainfall) & 
    (df_clean['delta_sm'] > 0)
].copy()


# 步骤 1: 只保留 RSM < 120 的数据（可选：也建议 RSM >= 0）
df_valid = df_valid[(df_valid['RSM'] >= 0) & (df_valid['RSM'] < 120)].copy()

# 步骤 2: 定义自定义 bins 和 labels
bins = [0, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
labels = [
    '0-30%',
    '30-40%',
    '40-50%',
    '50-60%',
    '60-70%',
    '70-80%',
    '80-90%',
    '90-100%',
    '100-110%',
    '110-120%'
]

# 步骤 3: 使用 pd.cut 分箱
df_valid['RSM_bin'] = pd.cut(
    df_valid['RSM'],
    bins=bins,
    labels=labels,
    include_lowest=True  # 确保 0 被包含在第一个 bin
)

# 可选：删除未落入任何 bin 的行（理论上不会出现，因已过滤 RSM<120 且 >=0）
df_valid = df_valid.dropna(subset=['RSM_bin']).reset_index(drop=True)

min_bin=df_valid['RSM_bin'].min()   
max_bin=df_valid['RSM_bin'].max()

min_bin_split=min_bin.split("-")[0]
max_bin_split=max_bin.split("-")[1]



# ---------------------------------------------------------
# 2. 分位数回归计算 TP_threshold
# ---------------------------------------------------------


# target_delta_sm_list=[10]
target_delta_sm = 10  # 有效湿化临界值 1%

# print(target_delta_sm)
results_data = []
bins = df_valid['RSM_bin'].unique()
tau = 0.50             # 分位数

all_qr_result=[]
# print(">>> 分位数回归 (QR) 计算结果:")
# print(f"{'Bin':<10} | {'Mean RSM':<10} | {'Intercept':<10} | {'Slope':<10} | {'TP_th (mm)':<10}")
# print("-" * 65)

for bin_label in bins:
    sub_df = df_valid[df_valid['RSM_bin'] == bin_label]
    
    col_index = sub_df.columns.get_loc('RSM_bin')
    RSM_bin = sub_df.iat[0, col_index]
    
    # QR 模型: delta_sm ~ TP
    mod = smf.quantreg('delta_sm ~ TP', sub_df)
    res = mod.fit(q=tau)
    
    beta_0 = res.params['Intercept']
    beta_1 = res.params['TP']
    
    # 计算阈值: TP_th = (1 - beta_0) / beta_1
    tp_th = (target_delta_sm - beta_0) / beta_1
    
    if 0 < tp_th < 100: # 计算RSM区间中点 
        mean_rsm = sub_df['RSM'].mean()
        
        results_data.append([mean_rsm, tp_th,RSM_bin])
        
        qr_results={
            "Bin":bin_label,
            "Mean RSM":round(mean_rsm,2),
            "Intercept":round(beta_0,4),
            "Slope":round(beta_1,4),
            "TP_th (mm)":round(tp_th,4)
            
            }
    
        all_qr_result.append(qr_results)
    
    
    # print(f"{bin_label:<10} | {mean_rsm:<10.2f} | {beta_0:<10.4f} | {beta_1:<10.4f} | {tp_th:<10.4f}")


qr_result_df=pd.DataFrame(all_qr_result)
# qr_result_df.to_csv(full_dir+'\\土壤分类{}湿度梯度增加{}QR回归结果1.csv'.format(joint_class,target_delta_sm),encoding='utf_8_sig', index=False, header=True)


# 转换为 DataFrame 用于拟合
df_fit = pd.DataFrame(results_data, columns=['x', 'y','RSM_bin'])
# 按 RSM 从小到大排序
df_fit = df_fit.sort_values(by='x')

# --- 注意 ---
# 提供的示例数据只有2个Bin点。2个点拟合2参数模型 R2 必然为 1，且无法计算P值（自由度不足）。
# 为了演示完整的代码逻辑，这里手动添加一个模拟点 (假设 RSM=45, 趋势下降)
# **实际使用时请删除下面这行代码，使用您的完整数据集**
if len(df_fit) < 3:
    print("\n[警告] 数据点不足3个，添加模拟点仅供演示代码功能...")
    df_fit.loc[len(df_fit)] = [45.0, 0.35] 

X = df_fit['x'].values
Y = df_fit['y'].values

# ---------------------------------------------------------
# 3. 曲线拟合与统计检验 (线性化方法)
# ---------------------------------------------------------

# A. 指数衰减模型: Y = a * e^(b * X)  =>  ln(Y) = ln(a) + b * X
#    令 Y' = ln(Y), A = ln(a), B = b
df_fit['ln_y'] = np.log(df_fit['y'])
exp_mod = smf.ols('ln_y ~ x', data=df_fit).fit()

exp_a = np.exp(exp_mod.params['Intercept'])
exp_b = exp_mod.params['x']
exp_r2 = exp_mod.rsquared
exp_p_val = exp_mod.pvalues['x']  # 斜率的显著性

# print(f"\n>>> 指数模型拟合结果 (Exponential):")
# print(f"Eq: TP_th = {exp_a:.4f} * e^({exp_b:.4f} * RSM)")
# print(f"R2: {exp_r2:.4f}, P-value: {exp_p_val:.4e}")

# B. 幂律模型: Y = a * X^b  =>  ln(Y) = ln(a) + b * ln(X)
#    令 Y' = ln(Y), X' = ln(X), A = ln(a), B = b
df_fit['ln_x'] = np.log(df_fit['x'])
pow_mod = smf.ols('ln_y ~ ln_x', data=df_fit).fit()

pow_a = np.exp(pow_mod.params['Intercept'])
pow_b = pow_mod.params['ln_x']
pow_r2 = pow_mod.rsquared
pow_p_val = pow_mod.pvalues['ln_x']

# print(f"\n>>> 幂律模型拟合结果 (Power Law):")
# print(f"Eq: TP_th = {pow_a:.4f} * RSM ^ ({pow_b:.4f})")
# print(f"R2: {pow_r2:.4f}, P-value: {pow_p_val:.4e}")

# ---------------------------------------------------------
# 2. 多项式回归模型 (Quadratic Model)
# 模型公式: Y = beta_0 + beta_1 * X + beta_2 * X^2
# ---------------------------------------------------------
# 使用 statsmodels 的公式接口，I(RSM**2) 表示 RSM 的平方项
poly_mod = smf.ols(formula='y ~ x + I(x**2)', data=df_fit).fit()

# 提取参数
beta_0 = poly_mod.params['Intercept']     # 截距 c
beta_1 = poly_mod.params['x']           # 一次项系数 b
beta_2 = poly_mod.params['I(x ** 2)']   # 二次项系数 a
r_squared = poly_mod.rsquared             # R2
p_values = poly_mod.pvalues               # P值

# 输出统计结果
# print(">>> 多项式模型 (二次) 拟合结果:")
# print(f"方程: y = {beta_2:.6f} * x^2 + {beta_1:.6f} * x + {beta_0:.6f}")
# print(f"R-squared (R2): {r_squared:.4f}")
# print(f"P-values:\n{p_values}")

muti_model_result={
    "Exp_Eq":f"TP_th = {exp_a:.4f} * e^({exp_b:.4f} * RSM)",
    "Exp_R2":f"{pow_r2:.4f}",
    "Exp_P-value":f"{pow_p_val:.4e}",
    "Pow_Eq":f"TP_th = {pow_a:.4f} * RSM ^ ({pow_b:.4f})",
    "Pow_R2":f"{pow_r2:.4f}",
    "Pow_P-value":f"{pow_p_val:.4e}",
    "Pol_Eq":f"y = {beta_2:.6f} * x^2 + {beta_1:.6f} * x + {beta_0:.6f}",
    "Pol_R2":f"{r_squared:.4f}",
    "Pol_P-value":f"{p_values[2]:.4f}",
    }

muti_result_df=pd.DataFrame([muti_model_result])
# muti_result_df.to_csv(full_dir+'\\土壤分类{}梯度湿度增加{}多模型拟合结果1.csv'.format(joint_class,target_delta_sm),encoding='utf_8_sig', index=False, header=True)


# ---------------------------------------------------------
# 4. 绘图代码
# ---------------------------------------------------------

# import matplotlib as mpl

# mpl.rcParams['font.family'] = 'Times New Roman'
# # mpl.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams['mathtext.rm'] = 'Times New Roman'
# mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
# mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'


# plt.rcParams["axes.unicode_minus"] = False  # 防止负号显示异常
# # 全局设置 Times New Roman 字体
# plt.rcParams['text.usetex'] = False  # 确保不使用 LaTeX（除非你有需求）


# plt.figure(figsize=(10, 6), dpi=100)

# 设置画布风格，类似你上传的图片（白色背景，无网格或淡网格）
# sns.set(style="ticks") 
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)



# B. 绘制散点 (蓝色实心点，带边框)
ax.scatter(X, Y,  color='black', s=80, edgecolors='black', zorder=5, label='Calculated Thresholds')


# 1. 绘制散点 (实测值)
# plt.scatter(X, Y, color='black', marker='o', s=80, label='Calculated Thresholds (QR, τ=0.25)', zorder=5)

# 2. 生成平滑曲线数据
x_smooth = np.linspace(X.min() * 0.95, X.max() * 1.05, 100)

# 计算拟合值
y_exp_smooth = exp_a * np.exp(exp_b * x_smooth)
y_pow_smooth = pow_a * np.power(x_smooth, pow_b)

# 代入二次方程计算 y
y_pol_smooth = beta_2 * (x_smooth**2) + beta_1 * x_smooth + beta_0


# 3. 绘制拟合曲线
# plt.plot(x_smooth, y_exp_smooth, 'g-', linewidth=2, label='Exponential Decay Fit')
# plt.plot(x_smooth, y_pow_smooth, 'b--', linewidth=2, label='Power Law Fit')
# plt.plot(x_smooth, y_pol_smooth, color='#e74c3c', linewidth=2, label='Polynomial Fit')

# A. 绘制拟合曲线 (红色虚线，模仿你的第一张图)
ax.plot(x_smooth, y_exp_smooth, 'g-', linewidth=2, label=f"Exponential Model: $R^2 = {exp_r2:.3f}, P = {exp_p_val:.3f}$")
ax.plot(x_smooth, y_pow_smooth, 'b--', linewidth=2, label=f"Power Model: $R^2 = {pow_r2:.3f}, P = {pow_p_val:.3f}$")
ax.plot(x_smooth, y_pol_smooth, color='#e74c3c', linewidth=2, label=f"Polynomial Model: $R^2 = {r_squared:.3f}, P = {p_values[2]:.3f}$")


# =========================================================
# [关键步骤] C. 添加文本标签
# =========================================================
# 遍历 DataFrame 的每一行
for index, row in df_fit.iterrows():
    # label_text = f"RSM {row['RSM_bin']} ({row['y']:.2f} mm)"
    label_text = f"{row['y']:.2f}"
    x_pos = row['x']
    y_pos = row['y']
    
    # 根据 index 是偶数还是奇数，决定位置
    if index % 2 == 0:
        # 右上：x 向右，y 向上；va='bottom' 表示文字底部在 (x,y) 上方 → 视觉在点上方
        dx, dy = 1.2, -0.03
        va = 'bottom'
    else:
        # 右下：x 向右，y 向下；va='top' 表示文字顶部在 (x,y) 下方 → 视觉在点下方
        dx, dy = 1.2, +0.03
        va = 'top'
    
    ax.text(
        x_pos + dx, 
        y_pos + dy, 
        label_text,
        color='black',
        fontsize=20,
        fontweight='normal',
        ha='left',      # 文字从 x_pos+dx 开始向右写（左侧对齐）
        va=va           # 垂直对齐方式：bottom（上）或 top（下）
    )
    
# === 设置刻度标签大小和加粗 ===
ax.tick_params(axis='both', which='major', labelsize=20, width=1, color='black')
ax.tick_params(axis='y', which='major', labelsize=20, width=1, color='black')

# 重新设置 y 轴标签以支持 fontweight
y_labels = [label.get_text() for label in ax.get_yticklabels()]
ax.set_yticklabels(y_labels, fontsize=20, fontweight='bold')

# 可选：x 轴标签也加粗
x_labels = [label.get_text() for label in ax.get_xticklabels()]
ax.set_xticklabels(x_labels, fontsize=20, fontweight='bold')
ax.xaxis.set_label_coords(0.5, -0.15)  # 调整 -0.15 到你想要的确切位置
 
# 4. 图表装饰
# plt.title('Precipitation threshold at a {}% increase in soil relative humidity gradient'.format(target_delta_sm), fontsize=14)
plt.xlabel('Soil Relative Humidity (%)', fontsize=22)
plt.ylabel('Precipitation Threshold (mm)', fontsize=22)
plt.grid(True, linestyle=':', alpha=0.6)

# 5. 添加统计信息文本框
# stats_text = (
#     f"Exponential Model:\n"
#     f"$y = {exp_a:.2f} e^{{{exp_b:.3f}x}}$\n"
#     f"$R^2 = {exp_r2:.3f}, P = {exp_p_val:.3f}$\n\n"
#     f"Power Law Model:\n"
#     f"$y = {pow_a:.2f} x^{{{pow_b:.3f}}}$\n"
#     f"$R^2 = {pow_r2:.3f}, P = {pow_p_val:.3f}$\n\n"
#     f"Polynomial Model:\n"
#     f"$y = {beta_2:.1e}x^2 + {beta_1:.3f}x {beta_0:.3f}$\n" # 使用科学计数法显示极小的二次项
#     f"$R^2 = {r_squared:.3f}, P = {p_values[2]:.3f}$"
   
# )

# # 放置文本框 (坐标根据数据范围自动调整，或者固定位置)
# box_props = dict(boxstyle='round', facecolor='white', alpha=0.9)
# plt.text(0.75, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
#          verticalalignment='top', bbox=box_props)

# zone_label = sector_map.get(fenqu_id, str(fenqu_id))  # 若不在字典中，回退为字符串数字

ax.text(0.95, 0.05, f'{joint_class}',
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        fontname='Times New Roman',
        verticalalignment='bottom',
        horizontalalignment='right',
        color='black')

# plt.legend( fontsize=10)
legend = ax.legend(fontsize=15)
plt.setp(legend.get_texts(), family='Times New Roman')
plt.tight_layout()

plt.savefig(full_dir+"\\土壤分类{}_{}-{}湿度梯度增加{}降水阈值1.png".format(joint_class,min_bin_split,max_bin_split,target_delta_sm), dpi=300, bbox_inches='tight')
plt.close() 
# 显示图形
# plt.show()        
        
