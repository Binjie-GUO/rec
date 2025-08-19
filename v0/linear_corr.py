# -*- coding: utf-8 -*-
"""
s3_11methods_linear.py
目标：
- 读取 ../data/oligo400_prediction_exp_n1-8.csv 与 ../data/Oligo400_total.csv
- 重现并补充线性关系验证（11种方法 + 专业可视化）
- 输出目录：analysis_results_log2_filtered
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset, het_white
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.graphics.gofplots import qqplot
import matplotlib

matplotlib.use('TkAgg')

# -------------------------
# ❶ 数据加载与预处理（路径保持不变）
# -------------------------
output_dir = "analysis_results_log2_filtered"
os.makedirs(output_dir, exist_ok=True)

# 1) 读取数据（路径保持不变）
file1 = pd.read_csv('../data_s/oligo400_prediction_exp_n1-8.csv')
file2 = pd.read_csv('../data_s/oligo400_total.csv')

# 2) 合并（以 iDNA 为 index）
merged_df = pd.merge(
    file2.set_index('iDNA'),
    file1,
    left_on='iAAs',
    right_on='iAAs',
    how='left'
)

print(f"原始数据行数: {len(merged_df)}")

# 3) 过滤
merged_df = merged_df.drop_duplicates(subset='iAAs', keep='first')
print(f"去除重复iAAs后行数: {len(merged_df)}")

filter_types = ["NEGATIVE SAMPLE", "POSITIVE SAMPLE", "Reported"]
filtered_df = merged_df[~merged_df['TYPE_x'].isin(filter_types)].copy()
print(f"过滤特定TYPE_x后行数: {len(filtered_df)}")

# 保存过滤后的数据
filtered_data_path = os.path.join(output_dir, 'filtered_data.csv')
filtered_df.to_csv(filtered_data_path, index=False)
print(f"已保存过滤后的数据: {filtered_data_path}")

# 4) 计算均值指标（沿用你原脚本的列命名规则）
filtered_df['LY6C1_mean'] = filtered_df[[c for c in file1.columns if ('LY6C1' in c and 'RPM' in c)]].mean(axis=1)
filtered_df['V_mean']     = filtered_df[[c for c in file1.columns if (c.startswith('V_') and 'RPM' in c)]].mean(axis=1)
filtered_df['LY6A_mean']  = filtered_df[[c for c in file1.columns if ('LY6A' in c and 'RPM' in c)]].mean(axis=1)
filtered_df['P_mean']     = filtered_df[[c for c in file1.columns if (c.startswith('P_') and 'RPM' in c)]].mean(axis=1)

# 5) log2 比值
filtered_df['LY6C1_V_log2'] = np.log2((filtered_df['LY6C1_mean']+1) / (filtered_df['V_mean']+1))
filtered_df['V_P_log2']     = np.log2((filtered_df['V_mean']+1)   / (filtered_df['P_mean']+1))
filtered_df['LY6A_V_log2']  = np.log2((filtered_df['LY6A_mean']+1)/ (filtered_df['V_mean']+1))

# 6) 脑富集均值 + log2
filtered_df['mean_brain_enrichment'] = (filtered_df['ENR_BRAIN1'] + filtered_df['ENR_BRAIN2']) / 2
filtered_df['log2_mean_brain_enrichment'] = np.log2(filtered_df['mean_brain_enrichment'] + 1)

# 保存计算后的数据
calculated_data_path = os.path.join(output_dir, 'calculated_data_log2.csv')
filtered_df.to_csv(calculated_data_path, index=False)
print(f"已保存计算后的数据: {calculated_data_path}")

# -------------------------
# ❷ 多方法线性关系验证（11种）
# -------------------------
def _valid_xy(df, x_col, y_col):
    tmp = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    x = tmp[x_col].values.reshape(-1, 1)
    y = tmp[y_col].values
    return tmp, x, y

def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def _ols_fit(x, y):
    X = sm.add_constant(x)  # [1, x]
    model = sm.OLS(y, X).fit()
    return model

def _ci_band_for_line(x, model):
    # OLS回归线95%置信带（针对每个观测x）
    X = sm.add_constant(x)
    pred = model.get_prediction(X)
    mean = pred.predicted_mean
    ci_low, ci_high = pred.conf_int().T  # ndarray
    return mean, ci_low, ci_high

def _piecewise_two_segments(x, y, grid_quantiles=(0.1, 0.9), n_grid=40):
    # 搜索单个拐点，左右两段线性
    q_low = np.quantile(x, grid_quantiles[0])
    q_high = np.quantile(x, grid_quantiles[1])
    grid = np.linspace(q_low, q_high, n_grid)
    best = None
    for c in grid:
        left = x <= c
        right = x > c
        if left.sum() < 5 or right.sum() < 5:
            continue
        X_left = sm.add_constant(x[left])
        X_right = sm.add_constant(x[right])
        m_left = sm.OLS(y[left], X_left).fit()
        m_right = sm.OLS(y[right], X_right).fit()
        y_pred = np.empty_like(y)
        y_pred[left] = m_left.predict(X_left)
        y_pred[right] = m_right.predict(X_right)
        rss = float(np.sum((y - y_pred) ** 2))
        if (best is None) or (rss < best["rss"]):
            best = {
                "c": float(c),
                "rss": rss,
                "model_left": m_left,
                "model_right": m_right,
                "y_pred": y_pred
            }
    return best

def _aic_bic_from_rss(n, k, rss):
    # AIC / BIC（高斯误差假设）
    sigma2 = rss / n
    ll = -0.5 * n * (np.log(2*np.pi*sigma2) + 1)
    aic = 2*k - 2*ll
    bic = np.log(n)*k - 2*ll
    return float(aic), float(bic)

def linearity_suite(df, x_col, y_col, prefix, outdir):
    """
    针对 x_col -> y_col 的线性关系，输出 11 种检验与可视化。
    返回：DataFrame（统计汇总），并写出 CSV 与多张图。
    """
    results = []
    tmp, X, y = _valid_xy(df, x_col, y_col)
    if len(tmp) < 10:
        print(f"[{prefix}] 有效样本不足（n={len(tmp)}），跳过。")
        return None

    x = X.squeeze()

    # 方法1：OLS + 95%CI
    ols_model = _ols_fit(x, y)
    slope = float(ols_model.params[1])
    intercept = float(ols_model.params[0])
    r2 = float(ols_model.rsquared)
    r2_adj = float(ols_model.rsquared_adj)
    slope_p = float(ols_model.pvalues[1])
    conf_int = np.asarray(ols_model.conf_int())  # 兼容 ndarray/DataFrame

    mean_line, ci_low, ci_high = _ci_band_for_line(x, ols_model)

    plt.figure(figsize=(7.5, 6))
    plt.scatter(x, y, alpha=0.6, s=40)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = intercept + slope*xs
    plt.plot(xs, ys, lw=2)
    order = np.argsort(x)
    plt.fill_between(x[order], ci_low[order], ci_high[order], alpha=0.2)
    plt.xlabel(x_col); plt.ylabel(y_col)
    plt.title(f"[1] OLS with 95% CI - {prefix}\n"
              f"slope={slope:.3f} (p={slope_p:.2e}), R²={r2:.3f}, adj.R²={r2_adj:.3f}")
    _savefig(os.path.join(outdir, f"{prefix}_01_OLS_CI.png"))

    results.append({
        "Method": "OLS",
        "Slope": slope, "Intercept": intercept,
        "Slope_p": slope_p, "R2": r2, "Adj_R2": r2_adj
    })

    # 方法2：LOWESS vs 直线
    low = lowess(y, x, return_sorted=True, frac=0.3)
    plt.figure(figsize=(7.5, 6))
    sns.scatterplot(x=x, y=y, alpha=0.5, s=40)
    plt.plot(xs, intercept + slope*xs, label="OLS", lw=2)
    plt.plot(low[:,0], low[:,1], label="LOWESS", lw=2)
    plt.xlabel(x_col); plt.ylabel(y_col)
    plt.title(f"[2] OLS vs LOWESS - {prefix}")
    plt.legend()
    _savefig(os.path.join(outdir, f"{prefix}_02_OLS_vs_LOWESS.png"))

    # 方法3：残差 vs 拟合值（含 LOWESS）
    fitted = ols_model.fittedvalues
    resid = ols_model.resid
    low_resid = lowess(resid, fitted, return_sorted=True, frac=0.3)
    plt.figure(figsize=(7.5, 6))
    plt.scatter(fitted, resid, alpha=0.6, s=40)
    plt.axhline(0, color='k', lw=1)
    plt.plot(low_resid[:,0], low_resid[:,1], lw=2)
    plt.xlabel("Fitted"); plt.ylabel("Residuals")
    plt.title(f"[3] Residuals vs Fitted (LOWESS) - {prefix}")
    _savefig(os.path.join(outdir, f"{prefix}_03_Residuals_vs_Fitted.png"))

    # 方法4：QQ图（残差正态性）
    plt.figure(figsize=(7.5, 6))
    qqplot(resid, line='45', fit=True)
    plt.title(f"[4] Residuals QQ-plot - {prefix}")
    _savefig(os.path.join(outdir, f"{prefix}_04_QQplot.png"))

    # 方法5：Scale-Location（同方差性）
    std_resid = (resid - resid.mean()) / resid.std(ddof=1)
    scl = np.sqrt(np.abs(std_resid))
    low_scl = lowess(scl, fitted, return_sorted=True, frac=0.3)
    plt.figure(figsize=(7.5, 6))
    plt.scatter(fitted, scl, alpha=0.6, s=40)
    plt.plot(low_scl[:,0], low_scl[:,1], lw=2)
    plt.xlabel("Fitted"); plt.ylabel("Sqrt(|Standardized Residuals|)")
    plt.title(f"[5] Scale-Location - {prefix}")
    _savefig(os.path.join(outdir, f"{prefix}_05_Scale_Location.png"))

    # 方法6：异方差检验（Breusch-Pagan / White）
    X_sm = sm.add_constant(x)
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, X_sm)
    white_stat, white_p, _, _ = het_white(resid, X_sm)
    results.append({"Method": "Breusch-Pagan", "Stat": float(bp_stat), "p_value": float(bp_p)})
    results.append({"Method": "White test", "Stat": float(white_stat), "p_value": float(white_p)})

    # 方法7：Ramsey RESET（遗漏非线性项检验）
    reset_res = linear_reset(ols_model, power=2, use_f=True)
    results.append({"Method": "Ramsey RESET (power=2)", "F": float(reset_res.fvalue), "p_value": float(reset_res.pvalue)})

    # 方法8：样条回归（cubic spline） vs 线性（AIC/BIC）
    try:
        import patsy
        from patsy import dmatrix
        knots = np.quantile(x, [0.25, 0.5, 0.75])
        X_spline = dmatrix("bs(x, knots=knots, degree=3, include_intercept=True)",
                           {"x": x, "knots": knots}, return_type='dataframe')
        spline_model = sm.OLS(y, X_spline).fit()

        # 线性/样条 AIC/BIC
        rss_lin = float(np.sum(ols_model.resid**2))
        aic_lin, bic_lin = _aic_bic_from_rss(len(y), 2, rss_lin)

        rss_spl = float(np.sum(spline_model.resid**2))
        aic_spl, bic_spl = _aic_bic_from_rss(len(y), X_spline.shape[1], rss_spl)

        xs_grid = np.linspace(x.min(), x.max(), 200)
        Xs_grid = dmatrix("bs(x, knots=knots, degree=3, include_intercept=True)",
                          {"x": xs_grid, "knots": knots}, return_type='dataframe')
        y_spl = spline_model.predict(Xs_grid)

        plt.figure(figsize=(7.5, 6))
        plt.scatter(x, y, alpha=0.5, s=40)
        plt.plot(xs, intercept + slope*xs, lw=2, label="Linear")
        plt.plot(xs_grid, y_spl, lw=2, label="Cubic spline")
        for k in knots:
            plt.axvline(k, ls='--', alpha=0.3)
        plt.xlabel(x_col); plt.ylabel(y_col)
        plt.title(f"[8] Spline vs Linear (AIC/BIC) - {prefix}\n"
                  f"Linear AIC={aic_lin:.1f}, BIC={bic_lin:.1f} | Spline AIC={aic_spl:.1f}, BIC={bic_spl:.1f}")
        plt.legend()
        _savefig(os.path.join(outdir, f"{prefix}_08_Spline_vs_Linear.png"))

        results.append({"Method": "Spline vs Linear (AIC/BIC)",
                        "Linear_AIC": aic_lin, "Linear_BIC": bic_lin,
                        "Spline_AIC": aic_spl, "Spline_BIC": bic_spl})
    except Exception as e:
        results.append({"Method": "Spline vs Linear (AIC/BIC)", "Error": str(e)})

    # 方法9：单拐点分段线性 vs 线性（AIC/BIC）
    seg = _piecewise_two_segments(x, y)
    if seg is not None:
        rss_seg = seg["rss"]
        # 近似自由度：两段各(截距+斜率)=4，自由度估计中忽略拐点 c 的额外参数以保持保守
        aic_seg, bic_seg = _aic_bic_from_rss(len(y), 4, rss_seg)

        plt.figure(figsize=(7.5, 6))
        plt.scatter(x, y, alpha=0.5, s=40)
        plt.plot(xs, intercept + slope*xs, lw=2, label="Linear")
        plt.axvline(seg["c"], ls='--', label=f"Breakpoint @ {seg['c']:.3f}")
        left = xs <= seg["c"]
        Xl = sm.add_constant(xs[left])
        Xr = sm.add_constant(xs[~left])
        yl = seg["model_left"].predict(Xl)
        yr = seg["model_right"].predict(Xr)
        plt.plot(xs[left], yl, lw=2, label="Segment Left")
        plt.plot(xs[~left], yr, lw=2, label="Segment Right")
        plt.xlabel(x_col); plt.ylabel(y_col)
        plt.title(f"[9] Piecewise Linear vs Linear (AIC/BIC) - {prefix}\n"
                  f"Piecewise AIC={aic_seg:.1f}, BIC={bic_seg:.1f}")
        plt.legend()
        _savefig(os.path.join(outdir, f"{prefix}_09_Piecewise_vs_Linear.png"))

        results.append({"Method": "Piecewise vs Linear (AIC/BIC)",
                        "Breakpoint": seg["c"], "Piecewise_AIC": aic_seg, "Piecewise_BIC": bic_seg})
    else:
        results.append({"Method": "Piecewise vs Linear (AIC/BIC)", "Note": "No valid breakpoint found"})

    # 方法10：Box-Cox（对 y 做变换以评估线性改善）
    y_pos = y.copy()
    min_y = y_pos.min()
    shift = 0.0
    if min_y <= 0:
        shift = abs(min_y) + 1e-6
        y_pos = y_pos + shift
    try:
        y_bc, lam = stats.boxcox(y_pos)
        model_bc = _ols_fit(x, y_bc)
        r2_bc = float(model_bc.rsquared)
        plt.figure(figsize=(7.5, 6))
        plt.scatter(x, y_bc, alpha=0.5, s=40)
        b0, b1 = float(model_bc.params[0]), float(model_bc.params[1])
        plt.plot(xs, b0 + b1*xs, lw=2)
        plt.xlabel(x_col); plt.ylabel("Box-Cox(y)")
        plt.title(f"[10] Box-Cox (λ={lam:.3f}, shift={shift:.2e}) - {prefix}\nR²={r2_bc:.3f}")
        _savefig(os.path.join(outdir, f"{prefix}_10_BoxCox.png"))
        results.append({"Method": "Box-Cox", "Lambda": float(lam), "Shift": float(shift), "R2_on_BoxCoxY": r2_bc})
    except Exception as e:
        results.append({"Method": "Box-Cox", "Error": str(e)})

    # 方法11：线性 vs 非线性（随机森林）5折 CV RMSE
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lin = LinearRegression()
    rf = RandomForestRegressor(n_estimators=300, random_state=42)

    lin_neg_mse = cross_val_score(lin, X, y, scoring='neg_mean_squared_error', cv=kf)
    rf_neg_mse  = cross_val_score(rf,  X, y, scoring='neg_mean_squared_error', cv=kf)

    lin_rmse = float(np.sqrt(-lin_neg_mse.mean()))
    rf_rmse  = float(np.sqrt(-rf_neg_mse.mean()))
    results.append({"Method": "CV RMSE (Linear vs RF)", "Linear_RMSE": lin_rmse, "RF_RMSE": rf_rmse})

    # 附加可视化A：Hexbin + 直线
    plt.figure(figsize=(7.5, 6))
    hb = plt.hexbin(x, y, gridsize=30, mincnt=1)
    plt.colorbar(hb, label='count')
    plt.plot(xs, intercept + slope*xs, lw=2, label="Linear")
    plt.xlabel(x_col); plt.ylabel(y_col)
    plt.title(f"[附] Hexbin + Linear - {prefix}")
    plt.legend()
    _savefig(os.path.join(outdir, f"{prefix}_A1_Hexbin.png"))

    # 附加可视化B：分箱残差
    bins = np.quantile(x, np.linspace(0, 1, 8))
    idx = np.digitize(x, bins[1:-1], right=True)
    df_bin = pd.DataFrame({"bin": idx, "fitted": fitted, "resid": resid})
    bin_avg = df_bin.groupby("bin")["resid"].mean()
    plt.figure(figsize=(7.5, 4))
    plt.plot(range(len(bin_avg)), bin_avg.values, marker='o')
    plt.axhline(0, color='k', lw=1)
    plt.xticks(range(len(bin_avg)), [f"B{i+1}" for i in range(len(bin_avg))])
    plt.ylabel("Mean Residual")
    plt.title(f"[附] Binned Residuals - {prefix}")
    _savefig(os.path.join(outdir, f"{prefix}_A2_BinnedResiduals.png"))

    # 附加可视化C：Cook's Distance
    infl = ols_model.get_influence()
    cooks = infl.cooks_distance[0]

    plt.figure(figsize=(7.5, 4))
    ax = plt.gca()

    # 兼容性：某些版本不支持 use_line_collection 参数
    import inspect
    stem_sig = inspect.signature(ax.stem)
    stem_kwargs = {}
    if "use_line_collection" in stem_sig.parameters:
        stem_kwargs["use_line_collection"] = True

    markerline, stemlines, baseline = ax.stem(np.arange(len(cooks)), cooks, **stem_kwargs)

    # 保险美化（不同版本返回对象略有差异，统一 try）
    try:
        plt.setp(markerline, markersize=4)
        plt.setp(stemlines, linewidth=1)
    except Exception:
        pass

    ax.set_title(f"[附] Cook's Distance - {prefix}")
    ax.set_xlabel("Observation Index")
    ax.set_ylabel("Cook's D")

    _savefig(os.path.join(outdir, f"{prefix}_A3_CooksDistance.png"))

    # 汇总结果
    res_df = pd.DataFrame(results)
    out_csv = os.path.join(outdir, f"{prefix}_linearity_suite_summary.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"[{prefix}] 线性关系验证完成，结果已保存：{out_csv}")
    return res_df

# -------------------------
# ❸ 对两个靶标分别执行
# -------------------------
targets = [
    ("LY6C1_V_log2", "log2_mean_brain_enrichment", "LY6C1_V_vs_Brain"),
    ("LY6A_V_log2", "log2_mean_brain_enrichment", "LY6A_V_vs_Brain")
]

all_summaries = []
for x_col, y_col, prefix in targets:
    if (x_col not in filtered_df.columns) or (y_col not in filtered_df.columns):
        print(f"跳过：缺少列 {x_col} 或 {y_col}")
        continue
    summary = linearity_suite(filtered_df, x_col, y_col, prefix, output_dir)
    if summary is not None:
        summary.insert(0, "Pair", prefix)
        all_summaries.append(summary)

if all_summaries:
    combined = pd.concat(all_summaries, ignore_index=True)
    combined_path = os.path.join(output_dir, "ALL_linearity_summaries.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n已保存所有线性验证方法的总览表：{combined_path}")

print("\n线性关系多方法验证：完成。输出目录：", output_dir)
