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
from adjustText import adjust_text
import matplotlib.patheffects as pe
#%%
# ---------------------------------------------------------
# 1. 数据准备 (基于您提供的 JSON 数据)
# ---------------------------------------------------------

joint_class='Loam_1'

full_dir = os.path.join(out_path1, joint_class)
# os.makedirs(full_dir)
# print(f"文件夹 '{full_dir}' 已创建。")

class_data = data_merge1[data_merge1['joint_class'] == joint_class]


#%%

fenqu_ids = sorted(class_data['zone'].unique())
fenqu_ids = [x for x in fenqu_ids if x != 5]

sector_map = {1: "A", 2: "B", 3: "C", 4: "D", 7: "F", 6: "G", 9: "H", 8: "I"}



for fenqu_id in fenqu_ids:
    print(f"\n=== 处理分区 {fenqu_id} ===")
    # fenqu_id=1.0

    
    # 提取当前分区数据
    fenqu_data = class_data[class_data['zone'] == int(fenqu_id)]
    fenqu_data_tiqu = fenqu_data[['zone', 'depth', 'TP', 'RSM', 'delta_sm']]
    

    
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
    
    depth_list=list(df_valid['depth'].unique())
    
    
    # === 全局字体设置（移到最前面，只需一次）===
    import matplotlib as mpl
    
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['mathtext.fontset'] = 'custom' 
    # plt.rcParams['font.weight'] = 'bold'  
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mpl.rcParams['mathtext.bf'] = 'Times New Roman'
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams['text.usetex'] = False
    
    # === 定义 depth 顺序和对应样式 ===
    depth_list = ['0-7cm', '7-28cm', '28-100cm']
    markers = {
        '0-7cm': 'o',
        '7-28cm': '^',
        '28-100cm': 's'
    }
    colors = {
        '0-7cm': 'red',
        '7-28cm': 'blue',
        '28-100cm': 'green'
    }
    
    # === 创建画布（只创建一次！）===
    # fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    fig, ax = plt.subplots(figsize=(16, 8))

    # 给右侧图例预留空间（关键！）
    # fig.subplots_adjust(right=0.72)
    # fig.subplots_adjust(top=0.80)

    # === 存储所有 df_fit 用于后续加标签（可选）===
    all_df_fit = []

    fenqu_qr_df=[]
    fenqu_model_df=[]

    for depth in depth_list:
        
        # depth='7-28cm'
        
        df_valid_tiqu=df_valid[df_valid['depth']==depth]
        
        # if df_valid_tiqu.empty:
        #     print(f"Warning: No data for depth {depth}")
        # continue
    
        max_tp=df_valid_tiqu['TP'].max()
    
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
        df_valid_tiqu['RSM_bin'] = pd.cut(
            df_valid_tiqu['RSM'],
            bins=bins,
            labels=labels,
            include_lowest=True  # 确保 0 被包含在第一个 bin
        )
        
        # 可选：删除未落入任何 bin 的行（理论上不会出现，因已过滤 RSM<120 且 >=0）
        df_valid_tiqu = df_valid_tiqu.dropna(subset=['RSM_bin']).reset_index(drop=True)
        
        min_bin=df_valid_tiqu['RSM_bin'].min()   
        max_bin=df_valid_tiqu['RSM_bin'].max()
        
        min_bin_split=min_bin.split("-")[0]
        max_bin_split=max_bin.split("-")[1]
        
        
        
        # ---------------------------------------------------------
        # 2. 分位数回归计算 TP_threshold
        # ---------------------------------------------------------
       
        
        # target_delta_sm_list=[1,2,3,4,5,6,7,8,9,10]
        target_delta_sm = 10  # 有效湿化临界值 1%
        
        # for target_delta_sm in target_delta_sm_list:
            
            # print(target_delta_sm)
        results_data = []
        bins = df_valid_tiqu['RSM_bin'].unique()
        tau = 0.5             # 分位数
        
        all_qr_result=[]
        # print(">>> 分位数回归 (QR) 计算结果:")
        # print(f"{'Bin':<10} | {'Mean RSM':<10} | {'Intercept':<10} | {'Slope':<10} | {'TP_th (mm)':<10}")
        # print("-" * 65)
        
        for bin_label in bins:
            sub_df = df_valid_tiqu[df_valid_tiqu['RSM_bin'] == bin_label]
            
                
            col_index = sub_df.columns.get_loc('RSM_bin')
            RSM_bin = sub_df.iat[0, col_index]
            
            # QR 模型: delta_sm ~ TP
            mod = smf.quantreg('delta_sm ~ TP', sub_df)
            res = mod.fit(q=tau)
            
            beta_0 = res.params['Intercept']
            beta_1 = res.params['TP']
            
            # 计算阈值: TP_th = (1 - beta_0) / beta_1
            tp_th = (target_delta_sm - beta_0) / beta_1
            # print(tp_th)
            
            if 0 < tp_th < max_tp: # 计算RSM区间中点 
                mean_rsm = sub_df['RSM'].mean()
                
                results_data.append([mean_rsm, tp_th,RSM_bin])
                
                qr_results={
                    "depth":depth,
                    "Bin":bin_label,
                    "Mean RSM":round(mean_rsm,2),
                    "Intercept":round(beta_0,4),
                    "Slope":round(beta_1,4),
                    "TP_th (mm)":round(tp_th,4)
                    
                    }
            
                all_qr_result.append(qr_results)
            
                # print(f"{bin_label:<10} | {mean_rsm:<10.2f} | {beta_0:<10.4f} | {beta_1:<10.4f} | {tp_th:<10.4f}")
            
        qr_result_df=pd.DataFrame(all_qr_result)
            
        fenqu_qr_df.append(qr_result_df)
        
            
            # 转换为 DataFrame 用于拟合
        df_fit = pd.DataFrame(results_data, columns=['x', 'y','RSM_bin'])
        # 按 RSM 从小到大排序
        df_fit = df_fit.sort_values(by='x')
        
        
        all_df_fit.append((depth, df_fit))  # 保存用于加标签
         
            
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
            "depth":depth,
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
        
        fenqu_model_df.append(muti_result_df)
        
        

        # ---------------------------------------------------------
        # 4. 绘图代码
        # ---------------------------------------------------------
        # fig, ax = plt.subplots(figsize=(16, 8))
        
        
        # 2. 生成平滑曲线数据
        x_smooth = np.linspace(X.min() * 0.95, X.max() * 1.05, 100)
        
        # 计算拟合值
        y_exp_smooth = exp_a * np.exp(exp_b * x_smooth)
        y_pow_smooth = pow_a * np.power(x_smooth, pow_b)
        
        # 代入二次方程计算 y
        y_pol_smooth = beta_2 * (x_smooth**2) + beta_1 * x_smooth + beta_0
        
        
        # ===============================
        # 6. 模型集合（核心）
        #    👉 优先 P → 再比 R²
        # ===============================
        model_pool = {
            'Exponential Model': {
                'p': exp_p_val,
                'r2': exp_r2,
                'y': y_exp_smooth
            },
            'Power Model': {
                'p': pow_p_val,
                'r2': pow_r2,
                'y': y_pow_smooth
            },
            'Polynomial Model': {
                'p': p_values[2],
                'r2': r_squared,
                'y': y_pol_smooth
            }
        }
        
        # ===============================
        # 7. 模型优选（P 最小 → R² 最大）
        # ===============================
        # best_model_name, best_model = sorted(
        #     model_pool.items(),
        #     key=lambda item: (item[1]['p'], -item[1]['r2'])
            
        # )[0]
        
        #优先 R² → 再比 P
        best_model_name, best_model = sorted(
            model_pool.items(),
            key=lambda item: (-item[1]['r2'], item[1]['p'])
        )[0]
        
        best_p  = best_model['p']
        best_r2 = best_model['r2']
        best_y  = best_model['y']

        # ===============================
        # 8. label_text 自动匹配模型类型
        # ===============================
        if best_p < 0.05:
            p_label = 'P<0.05'
        elif best_p < 0.1:
            p_label = 'P<0.1'
        else:
            p_label = ''
        
        # model_name_tex = best_model_name.replace(' ', r'~')

        # label_text = rf'$\mathbf{{{depth} : \mathrm{{{model_name_tex}}},\ R^2 = {best_r2:.3f}}}$'

        # label_text = rf'$\mathbf{{{depth} : \text{{{best_model_name}}},\ R^2 = {best_r2:.3f}}}$'
        label_text = rf'${depth} : \text{{{best_model_name}}},\ R^2 = {best_r2:.3f}$'

        # label_text = rf'$\mathbf{{{depth} : {best_model_name},\ R^2 = {best_r2:.3f}}}$'
        if p_label:
            label_text += f', {p_label}'
        
        # ===============================
        # 9. 绘图
        # ===============================
        
        ax.plot(
            x_smooth,
            best_y,
            color=colors[depth],
            linewidth=2,
            label=label_text,
            zorder=1
        )
        
        ax.scatter(
            X, Y,
            color='black',
            s=80,
            edgecolors='black',
            zorder=5,
            label='Calculated Thresholds (QR, τ=0.5)'
        )
        

    offset_map = {
        '0-7cm': (0, -8),
        '7-28cm': (0, 2),
        '28-100cm': (0, 2)
    }
    
    texts = []
    for depth, df_fit in all_df_fit:
        dx, dy = offset_map[depth]
        for _, row in df_fit.iterrows():
            texts.append(
                ax.text(
                    row['x'] + dx,
                    row['y'] + dy,
                    f"{row['y']:.2f}",
                    fontsize=18,
                    fontweight='bold',
                    zorder=10,
                    path_effects=[
                        pe.withStroke(linewidth=3, foreground='white')
                    ]
                )
            )

    
    adjust_text(
        texts,
        ax=ax,
        expand_points=(1.5, 3.5),
        expand_text=(1.5, 2.5),
        # arrowprops=dict(arrowstyle='-', lw=0.6, color='0.3'),
        force_points=1.2,
        force_text=1.0,
        autoalign=True,
        only_move={'points': 'y', 'text': 'y'}  # ❗只允许上下移动
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
    
    # === 图表装饰 ===
    # ax.set_title(f'Precipitation threshold at a {target_delta_sm}% increase in soil relative humidity gradient', fontsize=14)
    ax.set_xlabel('Soil Relative Humidity (%)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Precipitation Threshold (mm)', fontsize=20, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 区域标签（假设 fenqu_id, joint_class, sector_map 已定义）
    zone_label = sector_map.get(fenqu_id, str(fenqu_id))
    ax.text(
        0.2, 0.9, f'{joint_class} at zone {zone_label}',
        transform=ax.transAxes,
        fontsize=22,
        # fontweight='bold',
        fontname='Times New Roman',
        verticalalignment='bottom',
        horizontalalignment='right',
        color='black'
    )
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    legend = ax.legend(
        
        by_label.values(),
        by_label.keys(),
        prop={
        'family': 'Times New Roman',
        'size': 22,
        'weight': 'normal'
            },
        # fontsize=22,
        # fontweight='bold',
        ncol=2,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),   # 紧贴右侧
        borderaxespad=0.0,
        frameon=False
    )
    
    legend.get_frame().set_facecolor('none')  # 或者使用 set_alpha(0) 来达到相同效果
    
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')  # 确保字体正确
        # text.set_weight('bold')                # 强制加粗！
        text.set_fontweight('normal')
        # text.set_size(22)                      # 可选：统一字号
    
    
    plt.savefig(
    full_dir + "\\土壤分类{}分区{}_{}-{}湿度梯度增加{}降水阈值.png"
    .format(joint_class, fenqu_id, min_bin_split, max_bin_split, target_delta_sm),
    dpi=300,
    bbox_inches='tight'
                )
    
    plt.close()  

    fenqu_qr_df_concat=pd.concat(fenqu_qr_df)
    fenqu_mdoel_df_concat=pd.concat(fenqu_model_df)
    
    fenqu_qr_df_concat.to_csv(full_dir+'\\分区{}土壤分类{}湿度梯度增加{}QR回归结果.csv'.format(fenqu_id,joint_class,target_delta_sm),encoding='utf_8_sig', index=False, header=True)
    fenqu_mdoel_df_concat.to_csv(full_dir+'\\分区{}土壤分类{}梯度湿度增加{}多模型拟合结果.csv'.format(fenqu_id,joint_class,target_delta_sm),encoding='utf_8_sig', index=False, header=True)
        


df=class_data.copy()


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

depth_list=list(df_valid['depth'].unique())


# === 全局字体设置（移到最前面，只需一次）===
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom' 
# plt.rcParams['font.weight'] = 'bold'  
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman'
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['text.usetex'] = False

# === 定义 depth 顺序和对应样式 ===
depth_list = ['0-7cm', '7-28cm', '28-100cm']
markers = {
    '0-7cm': 'o',
    '7-28cm': '^',
    '28-100cm': 's'
}
colors = {
    '0-7cm': 'red',
    '7-28cm': 'blue',
    '28-100cm': 'green'
}

# === 创建画布（只创建一次！）===
# fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
fig, ax = plt.subplots(figsize=(18, 8))

# 给右侧图例预留空间（关键！）
# fig.subplots_adjust(right=0.72)
# fig.subplots_adjust(top=0.80)

# === 存储所有 df_fit 用于后续加标签（可选）===
all_df_fit = []

all_qr_df=[]
all_model_df=[]

for depth in depth_list:
    
    # depth='7-28cm'
    
    df_valid_tiqu=df_valid[df_valid['depth']==depth]
    
    # if df_valid_tiqu.empty:
    #     print(f"Warning: No data for depth {depth}")
    # continue

    max_tp=df_valid_tiqu['TP'].max()

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
    df_valid_tiqu['RSM_bin'] = pd.cut(
        df_valid_tiqu['RSM'],
        bins=bins,
        labels=labels,
        include_lowest=True  # 确保 0 被包含在第一个 bin
    )
    
    # 可选：删除未落入任何 bin 的行（理论上不会出现，因已过滤 RSM<120 且 >=0）
    df_valid_tiqu = df_valid_tiqu.dropna(subset=['RSM_bin']).reset_index(drop=True)
    
    min_bin=df_valid_tiqu['RSM_bin'].min()   
    max_bin=df_valid_tiqu['RSM_bin'].max()
    
    min_bin_split=min_bin.split("-")[0]
    max_bin_split=max_bin.split("-")[1]
    
    
    
    # ---------------------------------------------------------
    # 2. 分位数回归计算 TP_threshold
    # ---------------------------------------------------------
   
    
    # target_delta_sm_list=[1,2,3,4,5,6,7,8,9,10]
    target_delta_sm = 10  # 有效湿化临界值 1%
    
    # for target_delta_sm in target_delta_sm_list:
        
        # print(target_delta_sm)
    results_data = []
    bins = df_valid_tiqu['RSM_bin'].unique()
    tau = 0.5             # 分位数
    
    all_qr_result=[]
    # print(">>> 分位数回归 (QR) 计算结果:")
    # print(f"{'Bin':<10} | {'Mean RSM':<10} | {'Intercept':<10} | {'Slope':<10} | {'TP_th (mm)':<10}")
    # print("-" * 65)
    
    for bin_label in bins:
        sub_df = df_valid_tiqu[df_valid_tiqu['RSM_bin'] == bin_label]
        
            
        col_index = sub_df.columns.get_loc('RSM_bin')
        RSM_bin = sub_df.iat[0, col_index]
        
        # QR 模型: delta_sm ~ TP
        mod = smf.quantreg('delta_sm ~ TP', sub_df)
        res = mod.fit(q=tau)
        
        beta_0 = res.params['Intercept']
        beta_1 = res.params['TP']
        
        # 计算阈值: TP_th = (1 - beta_0) / beta_1
        tp_th = (target_delta_sm - beta_0) / beta_1
        # print(tp_th)
        
        if 0 < tp_th < max_tp: # 计算RSM区间中点 
            mean_rsm = sub_df['RSM'].mean()
            
            results_data.append([mean_rsm, tp_th,RSM_bin])
            
            qr_results={
                "depth":depth,
                "Bin":bin_label,
                "Mean RSM":round(mean_rsm,2),
                "Intercept":round(beta_0,4),
                "Slope":round(beta_1,4),
                "TP_th (mm)":round(tp_th,4)
                
                }
        
        all_qr_result.append(qr_results)
            
    qr_result_df=pd.DataFrame(all_qr_result)
        
    all_qr_df.append(qr_result_df)        
            # print(f"{bin_label:<10} | {mean_rsm:<10.2f} | {beta_0:<10.4f} | {beta_1:<10.4f} | {tp_th:<10.4f}")
        
        
            
        # 转换为 DataFrame 用于拟合
    df_fit = pd.DataFrame(results_data, columns=['x', 'y','RSM_bin'])
    # 按 RSM 从小到大排序
    df_fit = df_fit.sort_values(by='x')
    
    
    all_df_fit.append((depth, df_fit))  # 保存用于加标签
     
        
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
        "depth":depth,
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
    
    all_model_df.append(muti_result_df)
    

    # ---------------------------------------------------------
    # 4. 绘图代码
    # ---------------------------------------------------------
    # fig, ax = plt.subplots(figsize=(16, 8))
    
    
    # 2. 生成平滑曲线数据
    x_smooth = np.linspace(X.min() * 0.95, X.max() * 1.05, 100)
    
    # 计算拟合值
    y_exp_smooth = exp_a * np.exp(exp_b * x_smooth)
    y_pow_smooth = pow_a * np.power(x_smooth, pow_b)
    
    # 代入二次方程计算 y
    y_pol_smooth = beta_2 * (x_smooth**2) + beta_1 * x_smooth + beta_0
    
    
    # ===============================
    # 6. 模型集合（核心）
    #    👉 优先 P → 再比 R²
    # ===============================
    model_pool = {
        'Exponential Model': {
            'p': exp_p_val,
            'r2': exp_r2,
            'y': y_exp_smooth
        },
        'Power Model': {
            'p': pow_p_val,
            'r2': pow_r2,
            'y': y_pow_smooth
        },
        'Polynomial Model': {
            'p': p_values[2],
            'r2': r_squared,
            'y': y_pol_smooth
        }
    }
    
    # ===============================
    # 7. 模型优选（P 最小 → R² 最大）
    # ===============================
    # best_model_name, best_model = sorted(
    #     model_pool.items(),
    #     key=lambda item: (item[1]['p'], -item[1]['r2'])
        
    # )[0]
    
    #优先 R² → 再比 P
    best_model_name, best_model = sorted(
        model_pool.items(),
        key=lambda item: (-item[1]['r2'], item[1]['p'])
    )[0]
    
    best_p  = best_model['p']
    best_r2 = best_model['r2']
    best_y  = best_model['y']

    # ===============================
    # 8. label_text 自动匹配模型类型
    # ===============================
    if best_p < 0.05:
        p_label = 'P<0.05'
    elif best_p < 0.1:
        p_label = 'P<0.1'
    else:
        p_label = ''
    
   
    label_text = rf'${depth} : \text{{{best_model_name}}},\ R^2 = {best_r2:.3f}$'

    # label_text = rf'$\mathbf{{{depth} : {best_model_name},\ R^2 = {best_r2:.3f}}}$'
    if p_label:
        label_text += f', {p_label}'
    
    # ===============================
    # 9. 绘图
    # ===============================
    
    ax.plot(
        x_smooth,
        best_y,
        color=colors[depth],
        linewidth=2,
        label=label_text,
        zorder=1
    )
    
    ax.scatter(
        X, Y,
        color='black',
        s=80,
        edgecolors='black',
        zorder=5,
        label='Calculated Thresholds (QR, τ=0.5)'
    )
    


offset_map = {
    '0-7cm': (0, -8),
    '7-28cm': (0, 2),
    '28-100cm': (0, 2)
}

texts = []
for depth, df_fit in all_df_fit:
    dx, dy = offset_map[depth]
    for _, row in df_fit.iterrows():
        texts.append(
            ax.text(
                row['x'] + dx,
                row['y'] + dy,
                f"{row['y']:.2f}",
                fontsize=22,
                fontweight='bold',
                zorder=10,
                path_effects=[
                    pe.withStroke(linewidth=3, foreground='white')
                ]
            )
        )

adjust_text(
    texts,
    ax=ax,
    expand_points=(1.5, 3.5),
    expand_text=(1.5, 2.5),
    # arrowprops=dict(arrowstyle='-', lw=0.6, color='0.3'),
    force_points=1.2,
    force_text=1.0,
    autoalign=True,
    only_move={'points': 'y', 'text': 'y'}  # ❗只允许上下移动
)



# === 设置刻度标签大小和加粗 ===
ax.tick_params(axis='both', which='major', labelsize=25, width=1, color='black')
ax.tick_params(axis='y', which='major', labelsize=25, width=1, color='black')

# 重新设置 y 轴标签以支持 fontweight
y_labels = [label.get_text() for label in ax.get_yticklabels()]
ax.set_yticklabels(y_labels, fontsize=25, fontweight='bold')

# 可选：x 轴标签也加粗
x_labels = [label.get_text() for label in ax.get_xticklabels()]
ax.set_xticklabels(x_labels, fontsize=25, fontweight='bold')
ax.xaxis.set_label_coords(0.5, -0.15)  # 调整 -0.15 到你想要的确切位置

# === 图表装饰 ===
# ax.set_title(f'Precipitation threshold at a {target_delta_sm}% increase in soil relative humidity gradient', fontsize=14)
ax.set_xlabel('Soil Relative Humidity (%)', fontsize=25, fontweight='bold')
ax.set_ylabel('Precipitation Threshold (mm)', fontsize=25, fontweight='bold')
ax.grid(True, linestyle=':', alpha=0.6)

# 区域标签（假设 fenqu_id, joint_class, sector_map 已定义）
zone_label = sector_map.get(fenqu_id, str(fenqu_id))
ax.text(
    0.1, 0.9, f'{joint_class}',
    transform=ax.transAxes,
    fontsize=22,
    fontweight='bold',
    fontname='Times New Roman',
    verticalalignment='bottom',
    horizontalalignment='right',
    color='black'
)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))

legend = ax.legend(
    
    by_label.values(),
    by_label.keys(),
    prop={
    'family': 'Times New Roman',
    'size': 22,
    'weight': 'normal'
        },
    # fontsize=22,
    # fontweight='bold',
    ncol=2,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.25),   # 紧贴右侧
    borderaxespad=0.0,
    frameon=False
)

legend.get_frame().set_facecolor('none')  # 或者使用 set_alpha(0) 来达到相同效果

# 设置图例字体
for text in legend.get_texts():
    text.set_fontname('Times New Roman')  # 确保字体正确
    # text.set_weight('bold')                # 强制加粗！
    text.set_fontweight('normal')
    # text.set_size(22)                      # 可选：统一字号


plt.savefig(
full_dir + "\\土壤分类{}_{}-{}湿度梯度增加{}降水阈值.png"
.format(joint_class,  min_bin_split, max_bin_split, target_delta_sm),
dpi=300,
bbox_inches='tight'
            )

plt.close()          
    

all_qr_df_concat=pd.concat(all_qr_df)  
all_model_df_concat=pd.concat(all_model_df)  


        




