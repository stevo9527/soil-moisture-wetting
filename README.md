The code is for paper "An interpretable machine learning framework for disentangling driving factors and depth-dependent precipitation thresholds of soil wetting in China’s croplands". by Pingfan Fu, Xiaojing Yang, Dongya Sun , et al.

To appear in International Journal of Applied Earth Observation and Geoinformation 2026 written by Pingfan Fu. Email: fupf123456@163.com; yangxj@iwhr.com

Introductioin
Understanding soil wetting drivers is critical for water resources management, yet quantifying these mechanisms remains challenging due to complex spatial heterogeneity. This study proposes an interpretable machine learning framework, integrating eXtreme Gradient Boosting (XGBoost), Shapley Additive Explanations method (SHAP) and Quantile regression (QR), to disentangle daily-scale wetting dynamics from multi-source remote sensing data (2017–2022) across China’s agricultural regions. Subsequently, primary drivers and critical thresholds are identified across soil depths (0–100 cm), textures (loam, sandy loam, clay loam), and spatial regions (eight agricultural sub-zones). Results reveal a depth-dependent decoupling of wetting drivers where shallow layers (0–28 cm) are primarily governed by precipitation and antecedent moisture, whereas deep layers (28–100 cm) are regulated by the interplay of precipitation and evaporation. These drivers explain 54.6% and 29.3% of the variance in surface and deep layers, respectively. Additionally, a non-linear, U-shaped precipitation sensitivity curve was identified, delineating distinct state-dependent responsiveness zones. Efficient moisture conversion is maximized within specific Relative Soil Moisture (RSM) windows: 30–50%, 60–90%, and 100–120% for the surface (0–7 cm), intermediate (7–28 cm), and deep (28–100 cm) layers, respectively. Spatially, soil wetting initiation thresholds in northern agricultural zones exceed those in the south by 14.3%–20.9%, reflecting higher hydro-climatic resistance. Texture also modulates these thresholds, with sandy loam requiring 6.1%–7.4% higher precipitation inputs than finer textures. These findings provide a physical basis for optimizing macro-scale irrigation scheduling and enhancing the precision of hydrological early warning systems.


Requirements:
The code is tested on Windows 11 with Python 3.9.19.

Usage:
put pre-generated DI maps into the directory '\Data'. It is recommended to us log-ratio operator to genrate intial DI maps.
put their ground truth into the directory '\GT'.
run 'main.m'
