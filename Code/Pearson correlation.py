# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 08:34:46 2025

@author: fupf
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%%
"""
Correlation calculation
"""






filelist = []
for i in os.listdir(input_path):  # Traverse the entire folder
    #    print(i)
    path = os.path.join(input_path, i)
    filelist.append(path)
    
    
all_df=[]  
    
for file in filelist:
            
    data=pd.read_csv(file)
    
    file_name=os.path.basename(file)
    file_name=file_name.split(".")[0]
    
    # Initialize an empty DataFrame to store results
    correlation_df = pd.DataFrame(columns=['month', 
                                           '{}_pearson_corr'.format(file_name), 
                                           '{}_spearman_corr'.format(file_name), 
                                           '{}_kendall_corr'.format(file_name)])
     
    for month in range(1, 13):
        # Filter data for each month
        # month=1
        # month_data = all_merge_data_concat[all_merge_data_concat['month'] == month][[parameters[0], delta_sm_name]]
        month_data=data[data['month']==month]
        
        pearson_corr = round(month_data.iloc[:,3].corr(month_data.iloc[:,5]), 3)
        spearman_corr = round(month_data.iloc[:,3].corr(month_data.iloc[:,5], method='spearman'), 3)
        kendall_corr = round(month_data.iloc[:,3].corr(month_data.iloc[:,5], method='kendall'), 3)
        
        
        # Calculate correlation coefficient
        # pearson_corr = round(month_data[parameters[0]].corr(month_data[delta_sm_name]), 3)
        # spearman_corr = round(month_data[parameters[0]].corr(month_data[delta_sm_name], method='spearman'), 3)
        # kendall_corr = round(month_data[parameters[0]].corr(month_data[delta_sm_name], method='kendall'), 3)
        
        # Create a temporary DataFrame to store results for the current month
        temp_df = pd.DataFrame([{
            'month': month,
            '{}_pearson_corr'.format(file_name): pearson_corr,
            '{}_spearman_corr'.format(file_name): spearman_corr,
            '{}_kendall_corr'.format(file_name): kendall_corr
        }])
        
        # Append the temporary DataFrame to correlation_df
        correlation_df = pd.concat([correlation_df, temp_df], ignore_index=True)
        
        
    all_df.append(correlation_df)
 
    
pearson_corr_df=[]
for tmp_data in all_df:
    tmp_data_tiqu=tmp_data.iloc[:,1:2]
    
    pearson_corr_df.append(tmp_data_tiqu)

pearson_corr_df_concat= pd.concat(pearson_corr_df, axis=1)
pearson_corr_df_concat['month']=[1,2,3,4,5,6,7,8,9,10,11,12]
    
pearson_corr_df_concat.to_csv(out_path+'\\不同滞时pearson相关性结果.csv',encoding='utf_8_sig',index=None)
    # correlation_df.to_csv(out_path+'\\{}相关性数据.csv'.format(file_name),encoding='utf_8_sig',index=None)

#%%



'''
Correlation of response time after precipitation in different depths and zones
'''


# all_data=pd.read_csv(data_path)
quhua_data=pd.read_csv(quhua_path)

# all_data['lat'] = all_data['lat'].round(4)
# all_data['lon'] = all_data['lon'].round(4)
quhua_data['lat'] = quhua_data['lat'].round(4)
quhua_data['lon'] = quhua_data['lon'].round(4)

# all_data_quhua_merge=pd.merge(all_data,quhua_data,how='left',on=['lat','lon'])




filelist = []
for i in os.listdir(input_path):  # Traverse the entire folder
    #    print(i)
    path = os.path.join(input_path, i)
    filelist.append(path)
    
    
all_df=[]  
    
for file in filelist:
            
    data=pd.read_csv(file)
    
    data['lat'] = data['lat'].round(4)
    data['lon'] = data['lon'].round(4)
    quhua_data['lat'] = quhua_data['lat'].round(4)
    quhua_data['lon'] = quhua_data['lon'].round(4)

    data_quhua_merge=pd.merge(data,quhua_data,how='left',on=['lat','lon'])
    
    file_name=os.path.basename(file)
    file_name=file_name.split(".")[0]
    
    tmp_corr_df=[]
    
    for i in range(1,10):
        print(i)
        
        data_quhua_merge_tiqu=data_quhua_merge[data_quhua_merge['fenqu_ID']==i]
    
        # Initialize an empty DataFrame to store results
        correlation_df = pd.DataFrame(columns=[
                                               'month', 
                                               '{}_{}_pearson_corr'.format(file_name,i), 
                                               # '{}_{}_spearman_corr'.format(file_name,i), 
                                               # '{}_{}_kendall_corr'.format(file_name,i)
                                               ])
     
        for month in range(1, 13):
            # Filter data for each month
            # month=1
            # month_data = all_merge_data_concat[all_merge_data_concat['month'] == month][[parameters[0], delta_sm_name]]
            month_data=data_quhua_merge_tiqu[data_quhua_merge_tiqu['month']==month]
            
            pearson_corr = round(month_data.iloc[:,3].corr(month_data.iloc[:,5]), 3)
            # spearman_corr = round(month_data.iloc[:,3].corr(month_data.iloc[:,5], method='spearman'), 3)
            # kendall_corr = round(month_data.iloc[:,3].corr(month_data.iloc[:,5], method='kendall'), 3)
            
            
            # Calculate correlation coefficient
            # pearson_corr = round(month_data[parameters[0]].corr(month_data[delta_sm_name]), 3)
            # spearman_corr = round(month_data[parameters[0]].corr(month_data[delta_sm_name], method='spearman'), 3)
            # kendall_corr = round(month_data[parameters[0]].corr(month_data[delta_sm_name], method='kendall'), 3)
            
            # Create a temporary DataFrame to store results for the current month
            temp_df = pd.DataFrame([{
                'month': month,
                '{}_{}_pearson_corr'.format(file_name,i): pearson_corr,
                # '{}_{}_spearman_corr'.format(file_name,i): spearman_corr,
                # '{}_{}_kendall_corr'.format(file_name,i): kendall_corr
            }])
            
            # Append the temporary DataFrame to correlation_df
            correlation_df = pd.concat([correlation_df, temp_df], ignore_index=True)
            
            
        tmp_corr_df.append(correlation_df)
        
    all_tmp_corr_df=pd.concat(tmp_corr_df, axis=1)
    
    all_tmp_corr_df=all_tmp_corr_df.loc[:, ~all_tmp_corr_df.columns.duplicated()]
    
    all_df.append(all_tmp_corr_df)
     
     
all_df_concat=pd.concat(all_df,axis=1)     
all_df_concat=all_df_concat.loc[:, ~all_df_concat.columns.duplicated()]



tmp_data_tiqu=all_df_concat.iloc[:,1:]



# Calculate the median of each column
median_series = tmp_data_tiqu.median()

# Set global font to "Times New Roman"
from matplotlib import rcParams
from matplotlib.lines import Line2D     

   
 
letters = [chr(i) for i in range(ord('A'), ord('I') + 1)]  # A-I
numbers = ['1', '2', '3']
column_names = [l + n for n in numbers for l in letters]

tmp_data_tiqu.columns=column_names
# Grouping logic: first 1/3, middle 1/3, last 1/3 columns as a group
group_names = ['Group 1', 'Group 2', 'Group 3']
group_names = ['Lag 1 day', 'Lag 2 days', 'Lag 3 days']

columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# Create long format data
data_list = []

for col in columns:
    for group_idx, group_name in enumerate(group_names):
        cols = [f"{col}{group_idx + 1}"]
        subset = tmp_data_tiqu[cols]
        for val in subset.values.flatten():
            data_list.append({'Variable': col, 'Group': group_name, 'Value': val})

df_long = pd.DataFrame(data_list)




  
# === Set global font to Times New Roman ===
plt.rcParams['font.family'] = 'Times New Roman'
# ----------------------------

plt.figure(figsize=(15, 7))

ax = sns.boxplot(
    data=df_long,
    x='Variable',
    y='Value',
    hue='Group',
    palette='Set2',  # Use different colors
    patch_artist=True,  # Allow fill colors
    showmeans=True,     # ✅ Show mean (represented by marker)
    width=0.5,
    linewidth=1.5,
    flierprops=dict(marker='o', markersize=4, markeredgecolor='gray'),
    medianprops=dict(color='blue', linewidth=2),
    meanprops=dict(
        marker='^', 
        markerfacecolor='red', 
        markeredgecolor='red', 
        markersize=8,
        linewidth=2
    ),
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
)

# ----------------------------
# ✅ Key: Merge hue legend + custom legend
# ----------------------------

# # Get the hue legend automatically generated by seaborn
handles, labels = ax.get_legend_handles_labels()

# # Define custom legend elements (Median line, Average point)
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Median'),           # Median line
    Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
            markeredgecolor='red', markersize=8, linestyle='', label='Average')  # Mean point
]

# # Merge legend: first place Group grouping, then place statistics
combined_handles = handles + legend_elements
combined_labels = labels + [el.get_label() for el in legend_elements]

# Create final legend
ax.legend(
    handles=combined_handles,
    labels=combined_labels,
    loc='lower right',
    prop={'family': 'Times New Roman', 'size': 20},
    frameon=True,
    fancybox=False,
    edgecolor='black'
)

# ✅ Key: Remove legend (including hue automatically generated)
if ax.get_legend():
    ax.get_legend().remove()
    
    
# # --- Hide X-axis ticks and labels ---
# # Set X-axis tick labels to empty strings
# ax.set_xticklabels(['']*len(df_long['Variable'].unique()))
# # Or use:
# # plt.xticks([]) # Completely remove tick marks
# # If you want to keep tick lines but only hide labels, the above method is more applicable

# # Disable X-axis labels
# ax.set_xlabel('') 

# --- Set axis labels and tick fonts to "Times New Roman" ---
# X and Y axis tick labels font
plt.xticks(fontsize=20, fontfamily='Times New Roman', ha='right')
plt.yticks(fontsize=20, fontfamily='Times New Roman')

# Set axis label font
plt.xlabel('Agricultural zones', fontfamily='Times New Roman',fontsize=25)
plt.ylabel('Pearson correlation coefficient', fontfamily='Times New Roman',fontsize=25)
# plt.title('Boxplot of Pearson Correlation Coefficients by Variable Pair',
          # fontsize=20, fontfamily='Times New Roman')

# Enable grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Automatically adjust layout
plt.tight_layout()


plt.close()


#%%

"""
Draw boxplot
"""




correlation_df=pd.read_csv(data_path)

correlation_df=correlation_df.iloc[:,1:]


# Calculate the median of each column
median_series = correlation_df.median()
# Example data construction (please replace with your actual data)
# columns = [
#     'CHM_precipitation-ERA5_soil', 'CHM_precipitation-SMAP_soil', 'CHM_precipitation-SMCI_soil',
#     'ERA5_precipitation-ERA5_soil', 'ERA5_precipitation-SMAP_soil', 'ERA5_precipitation-SMCI_soil',
#     'CHIRPS_precipitation-ERA5_soil', 'CHIRPS_precipitation-SMAP_soil', 'CHIRPS_precipitation-SMCI_soil',
#     'GPM_precipitation-ERA5_soil', 'GPM_precipitation-SMAP_soil', 'GPM_precipitation-SMCI_soil'
# ]

columns = [
    'Combination1 lag 1d','Combination1 lag 2d','Combination1 lag 3d',
    'Combination2 lag 1d','Combination2 lag 2d','Combination2 lag 3d',
    'Combination3 lag 1d','Combination3 lag 2d','Combination3 lag 3d',
    'Combination4 lag 1d','Combination4 lag 2d','Combination4 lag 3d',
    'Combination5 lag 1d','Combination5 lag 2d','Combination5 lag 3d',
    'Combination6 lag 1d','Combination6 lag 2d','Combination6 lag 3d',
    'Combination7 lag 1d','Combination7 lag 2d','Combination7 lag 3d',
    'Combination8 lag 1d','Combination8 lag 2d','Combination8 lag 3d', 
    'Combination9 lag 1d','Combination9 lag 2d','Combination9 lag 3d',
]

# Set global font to "Times New Roman"
from matplotlib import rcParams
from matplotlib.lines import Line2D
# from matplotlib.font_manager import FontProperties



# ----------------------------

# Start plotting
plt.figure(figsize=(18, 10))

# Draw boxplot
boxplot = plt.boxplot(
    correlation_df.values,
    labels=columns,
    patch_artist=True,  # Allow fill colors
    showmeans=True,     # ✅ Show mean (represented by marker)
    showfliers=True,    # Optional: whether to show outliers
    medianprops=dict(color='blue', linewidth=2),  # Median line: blue, bold
    meanprops=dict(
        marker='^', 
        markerfacecolor='red', 
        markeredgecolor='red', 
        markersize=8,
        linewidth=2
    ),
    flierprops=dict(marker='o', markersize=4, markeredgecolor='gray'),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    boxprops=dict(linewidth=1.5)
)

# Add artists required for legend
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Median'), # Median line
    Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
           markeredgecolor='red', markersize=8, label='Average') # Mean point
]

# Show legend
plt.legend(handles=legend_elements, loc='upper right', prop={'family': 'Times New Roman', 'size': 20})

# --- Set color for each box ---
# You can use a single color, or set different colors for each box
colors = ['lightblue'] * len(correlation_df.columns)  # Uniform light blue
# Or use different colors:
# colors = plt.cm.viridis(np.linspace(0, 1, len(correlation_df.columns)))  # Gradient colors

for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')  # Border color
    patch.set_linewidth(1.5)      # Box border line width

# --- Add mean line (blue solid line) ---
means = correlation_df.mean().values
for i, mean in enumerate(means):
    plt.plot([i+1, i+1], [mean, mean], color='blue', linestyle='-', linewidth=2.5)

# --- Optional: Add standard deviation (red dashed line) ---
# stds = correlation_df.std().values
# for i, std in enumerate(stds):
#     plt.plot([i+1, i+1], [std, std], color='red', linestyle='--', linewidth=1.5)

# --- Optional: Add mean text ---
# for i, mean in enumerate(means):
#     plt.text(i + 1, mean - 0.05, f'{mean:.3f}', ha='center', va='top', fontsize=8, color='blue')

# --- Set axis labels and tick fonts to "Times New Roman" ---
# X and Y axis tick labels font
plt.xticks(fontsize=20, fontfamily='Times New Roman', rotation=45, ha='right')
plt.yticks(fontsize=20, fontfamily='Times New Roman')

# Set axis label font
plt.xlabel('Lag Days for Different Soil Moisture Responses to Precipitation', fontfamily='Times New Roman',fontsize=25)
plt.ylabel('Pearson correlation coefficient', fontfamily='Times New Roman',fontsize=25)
# plt.title('Boxplot of Pearson Correlation Coefficients by Variable Pair',
          # fontsize=20, fontfamily='Times New Roman')

# If you need more precise position control, you can use the following method:
ax = plt.gca()  # Get the current axes object
ax.yaxis.set_label_coords(-0.045, 0.5)  # Adjust -0.15 to the exact position you want

# Enable grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Automatically adjust layout
plt.tight_layout()


# plt.show()
plt.close()

# Display graphics
# plt.show()



"""
Draw correlation heatmap
"""


# Assume correlation_df is your DataFrame
# Example data creation (please replace with actual correlation_df when using)
# Set DataFrame index from 1 to 12

# === Set global font to Times New Roman ===
plt.rcParams['font.family'] = 'Times New Roman'

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
correlation_df.index=months

plt.figure(figsize=(20, 8))
sns.heatmap(correlation_df, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            xticklabels=columns, 
            yticklabels=correlation_df.index,
            annot_kws={"size": 16},      # Adjust the font size of numbers in the grid
            cbar_kws={"shrink": 0.8},
            square=False                 # Do not force square grids, save space
            )

# Set x-axis ticks to 45°
# plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(rotation=360, fontsize=20)

# ✅ Key: Hide x-axis ticks and labels
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)

# plt.xlabel('Precipitation-Δsoil moisture', fontsize=16)
plt.ylabel('Monthly Pearson correlation coefficient', fontsize=25)

# plt.title('Heatmap of Pearson Correlation Coefficients', fontsize=16, fontfamily='Times New Roman')
plt.tight_layout()

# plt.show()
plt.close()
