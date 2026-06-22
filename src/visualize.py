"""
visualize.py
图表生成模块，封装全部 12 张可视化函数

  基础图表: fig01~fig07 (趋势/对比/箱线/散点/热力/相关/直方)
  高级图表: fig08~fig12 (交互/异常标注/ANOVA/KMeans/回归)
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 全局配色体系 =====
# 健康状态语义色
STATUS_PALETTE = {'健康': '#59A96C', '中度胁迫': '#E5A93D', '高度胁迫': '#E0555A'}

# 指标语义色（跨图表一致）
METRIC_COLORS = {
    'Ambient_Temperature': '#E06C75',   # 珊瑚红 — 温度/暖
    'Humidity':            '#61AFEF',   # 天空蓝 — 湿度/水
    'Soil_Moisture':       '#7EC87B',   # 草绿   — 土壤
    'Light_Intensity':     '#D4A843',   # 琥珀金 — 光照
    'Soil_Temperature':    '#E8965B',   # 橘棕   — 土温
    'Nitrogen_Level':      '#C678DD',   # 丁香紫 — 氮
    'Chlorophyll_Content': '#4DB89E',   # 青绿   — 叶绿素
    'Soil_pH':             '#9AACB8',   # 灰蓝   — pH
}

# 植株 10 色板（彩色盲友好）
PLANT_COLORS = ['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974',
                '#64B5CD','#E89243','#7EC87B','#C678DD','#8C8C8C']

# 聚类 3 色
CLUSTER_COLORS = ['#E06C75', '#61AFEF', '#7EC87B']

OUTPUT_DIR = '../outputs/figures/'


def fig00_overview(df_clean: pd.DataFrame) -> None:
    """图0：数据概览面板 — 健康状态饼图 + 关键指标对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('茶园植物健康环境监测 — 数据概览', fontsize=14, fontweight='bold')

    health_counts = df_clean['健康状态'].value_counts()
    colors_pie = [STATUS_PALETTE[k] for k in health_counts.index]
    wedges, texts, autotexts = ax1.pie(
        health_counts.values, labels=health_counts.index, autopct='%1.1f%%',
        colors=colors_pie, explode=(0.03, 0.03, 0.03),
        textprops={'fontsize': 12}, pctdistance=0.6
    )
    for at in autotexts:
        at.set_fontweight('bold'); at.set_fontsize(11)
    ax1.set_title('植物健康状态分布', fontsize=14, fontweight='bold')

    healthy = df_clean[df_clean['健康状态'] == '健康']
    stressed = df_clean[df_clean['健康状态'] == '高度胁迫']
    key_metrics = ['Soil_Moisture', 'Nitrogen_Level', 'Ambient_Temperature', 'Humidity', 'Chlorophyll_Content']
    key_labels = ['土壤湿度\n(%)', '氮含量\n(mg/kg)', '环境温度\n(℃)', '空气湿度\n(%)', '叶绿素\n(SPAD)']
    x = np.arange(len(key_labels)); w = 0.35
    ax2.bar(x - w/2, [healthy[m].mean() for m in key_metrics], w, color=STATUS_PALETTE['健康'], label='健康', edgecolor='white')
    ax2.bar(x + w/2, [stressed[m].mean() for m in key_metrics], w, color=STATUS_PALETTE['高度胁迫'], label='高度胁迫', edgecolor='white')
    ax2.set_xticks(x); ax2.set_xticklabels(key_labels, fontsize=9)
    ax2.set_ylabel('均值', fontsize=11)
    ax2.set_title('健康 vs 高度胁迫 关键指标对比', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10); ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig00_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图0（概览） 保存完成')


def fig01_daily_trends(df_clean: pd.DataFrame) -> None:
    """图1：各指标日均变化趋势折线图"""
    daily = df_clean.groupby('date')[
        ['Ambient_Temperature', 'Humidity', 'Soil_Moisture', 'Light_Intensity']
    ].mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('茶园植物健康环境监测 - 各指标日均变化趋势', fontsize=15, fontweight='bold')

    configs = [
        ('Ambient_Temperature', '环境温度 (℃)', axes[0, 0], METRIC_COLORS['Ambient_Temperature']),
        ('Humidity', '空气湿度 (%)', axes[0, 1], METRIC_COLORS['Humidity']),
        ('Soil_Moisture', '土壤湿度 (%)', axes[1, 0], METRIC_COLORS['Soil_Moisture']),
        ('Light_Intensity', '光照强度 (lux)', axes[1, 1], METRIC_COLORS['Light_Intensity']),
    ]
    for col, ylabel, ax, color in configs:
        ax.plot(range(len(daily)), daily[col], color=color, linewidth=2, marker='o', markersize=4)
        ax.fill_between(range(len(daily)), daily[col], alpha=0.15, color=color)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel('监测天数', fontsize=10)
        ax.set_title(ylabel.split(' ')[0] + '日均变化', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig01_daily_trends.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图1 保存完成')


def fig02_plant_comparison(df_clean: pd.DataFrame) -> None:
    """图2：各监测植株环境指标对比条形图"""
    plant_avg = df_clean.groupby('Plant_ID_str')[
        ['Ambient_Temperature', 'Humidity', 'Soil_Moisture']
    ].mean()

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('各监测植株环境指标对比', fontsize=15, fontweight='bold')

    for i, (col, ylabel) in enumerate([
        ('Ambient_Temperature', '环境温度 (℃)'),
        ('Humidity', '空气湿度 (%)'),
        ('Soil_Moisture', '土壤湿度 (%)'),
    ]):
        bars = axes[i].bar(range(len(plant_avg)), plant_avg[col], color=PLANT_COLORS, edgecolor='white')
        axes[i].set_xticks(range(len(plant_avg)))
        axes[i].set_xticklabels(plant_avg.index, rotation=45, fontsize=8)
        axes[i].set_title(ylabel.split(' ')[0] + '对比', fontsize=12)
        axes[i].set_ylabel(ylabel, fontsize=10)
        axes[i].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, plant_avg[col]):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig02_plant_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图2 保存完成')


def fig03_health_boxplot(df_clean: pd.DataFrame) -> None:
    """图3：不同健康状态温度与土壤湿度箱线图"""
    order = ['健康', '中度胁迫', '高度胁迫']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('不同健康状态下温度与土壤湿度分布', fontsize=14, fontweight='bold')

    sns.boxplot(data=df_clean, x='健康状态', y='Ambient_Temperature',
                order=order, palette=STATUS_PALETTE, ax=axes[0], width=0.5)
    axes[0].set_title('不同健康状态下环境温度分布', fontsize=12)
    axes[0].set_xlabel('健康状态'); axes[0].set_ylabel('环境温度 (℃)'); axes[0].grid(axis='y', alpha=0.3)

    sns.boxplot(data=df_clean, x='健康状态', y='Soil_Moisture',
                order=order, palette=STATUS_PALETTE, ax=axes[1], width=0.5)
    axes[1].set_title('不同健康状态下土壤湿度分布', fontsize=12)
    axes[1].set_xlabel('健康状态'); axes[1].set_ylabel('土壤湿度 (%)'); axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig03_health_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图3 保存完成')


def fig08_interactive(df_clean: pd.DataFrame) -> None:
    """图8：温度与湿度逐日变化趋势（上下分开展示，避免双Y轴混淆）"""
    daily = df_clean.groupby('date')[
        ['Ambient_Temperature', 'Humidity']
    ].mean().reset_index()
    daily['date'] = daily['date'].astype(str)
    x_ticks = range(0, len(daily), max(1, len(daily) // 10))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('茶园环境温度与空气湿度日均变化趋势', fontsize=15, fontweight='bold')

    # 上图：温度
    temp_color = METRIC_COLORS['Ambient_Temperature']
    ax1.plot(range(len(daily)), daily['Ambient_Temperature'],
             color=temp_color, linewidth=2, marker='o', markersize=3)
    ax1.fill_between(range(len(daily)), daily['Ambient_Temperature'],
                     alpha=0.12, color=temp_color)
    ax1.set_ylabel('环境温度 (℃)', fontsize=12, color=temp_color)
    ax1.set_title('环境温度逐日变化', fontsize=13)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([daily['date'].iloc[i] for i in x_ticks], rotation=30, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 标注最高温、最低温
    t_max_idx = daily['Ambient_Temperature'].idxmax()
    t_min_idx = daily['Ambient_Temperature'].idxmin()
    ax1.annotate(f'最高 {daily["Ambient_Temperature"].loc[t_max_idx]:.1f}℃',
                 xy=(t_max_idx, daily['Ambient_Temperature'].loc[t_max_idx]),
                 xytext=(t_max_idx - 3, daily['Ambient_Temperature'].loc[t_max_idx] + 0.3),
                 fontsize=9, color='#C44E52', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#C44E52'))
    ax1.annotate(f'最低 {daily["Ambient_Temperature"].loc[t_min_idx]:.1f}℃',
                 xy=(t_min_idx, daily['Ambient_Temperature'].loc[t_min_idx]),
                 xytext=(t_min_idx + 2, daily['Ambient_Temperature'].loc[t_min_idx] - 0.5),
                 fontsize=9, color='#4C72B0', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#4C72B0'))

    # 下图：湿度
    hum_color = METRIC_COLORS['Humidity']
    ax2.plot(range(len(daily)), daily['Humidity'],
             color=hum_color, linewidth=2, marker='o', markersize=3)
    ax2.fill_between(range(len(daily)), daily['Humidity'],
                     alpha=0.12, color=hum_color)
    ax2.set_ylabel('空气湿度 (%)', fontsize=12, color=hum_color)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_title('空气湿度逐日变化', fontsize=13)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([daily['date'].iloc[i] for i in x_ticks], rotation=30, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 标注最高湿、最低湿
    h_max_idx = daily['Humidity'].idxmax()
    h_min_idx = daily['Humidity'].idxmin()
    ax2.annotate(f'最高 {daily["Humidity"].loc[h_max_idx]:.1f}%',
                 xy=(h_max_idx, daily['Humidity'].loc[h_max_idx]),
                 xytext=(h_max_idx - 3, daily['Humidity'].loc[h_max_idx] + 0.8),
                 fontsize=9, color='#2878A0', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#2878A0'))
    ax2.annotate(f'最低 {daily["Humidity"].loc[h_min_idx]:.1f}%',
                 xy=(h_min_idx, daily['Humidity'].loc[h_min_idx]),
                 xytext=(h_min_idx + 2, daily['Humidity'].loc[h_min_idx] + 1.0),
                 fontsize=9, color='#E5A93D', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#E5A93D'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig08_interactive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图8 保存完成')


def fig10_anova(df_clean: pd.DataFrame) -> pd.DataFrame:
    """图10：各健康状态下核心指标对比（含ANOVA显著性标注）—— 上下分面，避免光照强度碾压其他指标"""
    # 拆两组，各有独立Y轴
    top_cols = ['Ambient_Temperature', 'Soil_Moisture', 'Humidity']
    top_labels = ['环境温度\n(℃)', '土壤湿度\n(%)', '空气湿度\n(%)']

    bottom_cols = ['Light_Intensity', 'Nitrogen_Level', 'Chlorophyll_Content']
    bottom_labels = ['光照强度\n(lux)', '氮含量\n(mg/kg)', '叶绿素\n(SPAD)']

    all_cols = top_cols + bottom_cols
    all_labels = top_labels + bottom_labels

    # 计算三组均值 + ANOVA
    means = {}
    f_vals, p_vals = [], []
    for col in all_cols:
        grp = df_clean.groupby('健康状态')[col].mean()
        means[col] = {'健康': grp.get('健康', 0),
                      '中度胁迫': grp.get('中度胁迫', 0),
                      '高度胁迫': grp.get('高度胁迫', 0)}
        groups = [df_clean[df_clean['健康状态'] == s][col].dropna()
                  for s in ['健康', '中度胁迫', '高度胁迫']]
        f, p = stats.f_oneway(*groups)
        f_vals.append(f); p_vals.append(p)

    def sig_label(p):
        if p < 0.001: return '极显著'
        if p < 0.01: return '很显著'
        if p < 0.05: return '显著'
        return '不显著'

    results_df = pd.DataFrame({
        '指标': [l.replace('\n', '') for l in all_labels],
        'F值': [round(f, 2) for f in f_vals],
        'p值': [f'{p:.4f}' if p >= 0.0001 else '<0.0001' for p in p_vals],
        '显著性': [sig_label(p) for p in p_vals]
    })

    # 上下两个子图，各有独立Y轴
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('不同健康状态下各环境与营养指标均值对比', fontsize=16, fontweight='bold')

    width = 0.25
    status_order = ['健康', '中度胁迫', '高度胁迫']
    bar_colors = [STATUS_PALETTE[s] for s in status_order]

    for ax, cols, labels in [(ax1, top_cols, top_labels),
                              (ax2, bottom_cols, bottom_labels)]:
        x = np.arange(len(labels))

        for j, (status, color) in enumerate(zip(status_order, bar_colors)):
            offset = (j - 1) * width
            bars = ax.bar(x + offset, [means[c][status] for c in cols],
                          width, color=color, label=status, edgecolor='white')
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f'{h:.1f}', ha='center', va='bottom', fontsize=8)

        # 显著性标注 —— 每个指标用自己的柱高
        for i, col in enumerate(cols):
            p = p_vals[all_cols.index(col)]
            y_pos = max(means[col].values()) * 1.12
            label = sig_label(p)
            c = '#E0555A' if p < 0.05 else '#9AACB8'
            fw = 'bold' if p < 0.05 else 'normal'
            ax.text(i, y_pos, label, ha='center', fontsize=12, color=c, fontweight=fw)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('均值', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    # 底部图例说明
    ax2.text(0.5, -0.15,
             '柱顶标注：极显著 = p<0.001 | 很显著 = p<0.01 | 显著 = p<0.05 | 不显著 = 无统计学差异',
             transform=ax2.transAxes, fontsize=9, ha='center', color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig10_anova.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图10 保存完成')
    return results_df


def fig11_kmeans(df_clean: pd.DataFrame) -> None:
    """图11：K-Means聚类分析"""
    cluster_cols = ['Ambient_Temperature', 'Humidity', 'Soil_Moisture',
                    'Light_Intensity', 'Nitrogen_Level', 'Chlorophyll_Content']
    X = df_clean[cluster_cols].dropna()
    X_scaled = StandardScaler().fit_transform(X)

    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_
                for k in range(2, 8)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('植株环境指标K-Means聚类分析', fontsize=14, fontweight='bold')

    axes[0].plot(range(2, 8), inertias, color='#5D8AA8', marker='o', linewidth=2, markersize=8,
                 markerfacecolor='#4C72B0', markeredgecolor='white', markeredgewidth=1)
    axes[0].axvline(x=3, color='#E0555A', linestyle='--', linewidth=1.5, label='选定K=3')
    axes[0].set_xlabel('聚类数量K', fontsize=11); axes[0].set_ylabel('簇内误差平方和', fontsize=11)
    axes[0].set_title('肘部法则确定最优K值', fontsize=12)
    axes[0].legend(); axes[0].grid(alpha=0.3)

    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km3.fit_predict(X_scaled)
    df_cluster = df_clean.loc[X.index].copy()
    df_cluster['聚类'] = [f'群组{l + 1}' for l in labels]

    cluster_colors = {'群组1': CLUSTER_COLORS[0], '群组2': CLUSTER_COLORS[1], '群组3': CLUSTER_COLORS[2]}
    for cluster, group in df_cluster.groupby('聚类'):
        axes[1].scatter(group['Ambient_Temperature'], group['Chlorophyll_Content'],
                        c=cluster_colors[cluster], label=cluster, alpha=0.6, s=30, edgecolors='none')
    axes[1].set_xlabel('环境温度 (℃)', fontsize=11); axes[1].set_ylabel('叶绿素含量 (SPAD)', fontsize=11)
    axes[1].set_title('K-Means聚类结果（温度 vs 叶绿素）', fontsize=12)
    axes[1].legend(title='聚类群组'); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig11_kmeans.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图11 保存完成')


def fig12_regression(df_clean: pd.DataFrame) -> None:
    """图12：光照强度对叶绿素含量线性回归"""
    X_reg = df_clean[['Light_Intensity']].values
    y_reg = df_clean['Chlorophyll_Content'].values
    reg = LinearRegression().fit(X_reg, y_reg)
    y_pred = reg.predict(X_reg)
    r2 = r2_score(y_reg, y_pred)
    residuals = y_reg - y_pred

    x_line = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
    y_line = reg.predict(x_line)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('光照强度对叶绿素含量线性回归分析', fontsize=14, fontweight='bold')

    axes[0].scatter(X_reg, y_reg, alpha=0.25, s=18, color='#4C72B0', edgecolors='none', label='观测值')
    axes[0].plot(x_line, y_line, color='#E0555A', linewidth=2.5,
                 label=f'回归线: y={reg.coef_[0]:.4f}x+{reg.intercept_:.2f}')
    ci = 1.96 * np.std(residuals)
    axes[0].fill_between(x_line.flatten(), y_line - ci, y_line + ci,
                         alpha=0.12, color='#E0555A', label='95%置信区间')
    axes[0].set_xlabel('光照强度 (lux)', fontsize=11); axes[0].set_ylabel('叶绿素含量 (SPAD)', fontsize=11)
    axes[0].set_title(f'线性回归 (R$^2$={r2:.4f})', fontsize=12)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].scatter(y_pred, residuals, alpha=0.25, s=18, color='#8172B2', edgecolors='none')
    axes[1].axhline(0, color='#E0555A', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('预测值', fontsize=11); axes[1].set_ylabel('残差', fontsize=11)
    axes[1].set_title('残差分布图', fontsize=12); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig12_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'图12 保存完成 | R²={r2:.4f}')


def fig04_light_chlorophyll(df_clean: pd.DataFrame) -> None:
    """图4：光照强度与叶绿素含量相关性散点图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    for status, group in df_clean.groupby('Plant_Health_Status'):
        label_map = {'Healthy': '健康', 'Moderate Stress': '中度胁迫', 'High Stress': '高度胁迫'}
        color = STATUS_PALETTE.get(label_map.get(status, status), 'gray')
        ax.scatter(group['Light_Intensity'], group['Chlorophyll_Content'],
                   c=color, label=label_map.get(status, status),
                   alpha=0.5, s=25, edgecolors='none')

    z = np.polyfit(df_clean['Light_Intensity'], df_clean['Chlorophyll_Content'], 1)
    p = np.poly1d(z)
    xl = np.linspace(df_clean['Light_Intensity'].min(), df_clean['Light_Intensity'].max(), 100)
    ax.plot(xl, p(xl), 'k--', linewidth=2, label='趋势线')

    corr = df_clean['Light_Intensity'].corr(df_clean['Chlorophyll_Content'])
    ax.set_title(f'光照强度与叶绿素含量相关性分析 (r={corr:.3f})', fontsize=13, fontweight='bold')
    ax.set_xlabel('光照强度 (lux)')
    ax.set_ylabel('叶绿素含量 (SPAD)')
    ax.legend(title='健康状态')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig04_light_chlorophyll.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图4 保存完成')


def fig05_heatmap(df_clean: pd.DataFrame) -> None:
    """图5：各植株逐日温度与湿度热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('各植株逐日温度与湿度热力图', fontsize=14, fontweight='bold')

    pt = df_clean.groupby(['Plant_ID_str', 'day'])['Ambient_Temperature'].mean().unstack()
    ph = df_clean.groupby(['Plant_ID_str', 'day'])['Humidity'].mean().unstack()

    sns.heatmap(pt, cmap='YlOrRd', ax=axes[0], linewidths=0.3,
                cbar_kws={'label': '温度 (℃)'}, annot=False)
    axes[0].set_title('各植株逐日环境温度 (℃)', fontsize=12)
    axes[0].set_xlabel('日期（天）')
    axes[0].set_ylabel('植株编号')

    sns.heatmap(ph, cmap='YlGnBu', ax=axes[1], linewidths=0.3,
                cbar_kws={'label': '湿度 (%)'}, annot=False)
    axes[1].set_title('各植株逐日空气湿度 (%)', fontsize=12)
    axes[1].set_xlabel('日期（天）')
    axes[1].set_ylabel('植株编号')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig05_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图5 保存完成')


def fig06_correlation(df_clean: pd.DataFrame) -> None:
    """图6：各指标相关性矩阵热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    corr_cols = ['Ambient_Temperature', 'Soil_Temperature', 'Humidity', 'Soil_Moisture',
                 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', 'Chlorophyll_Content']
    corr_labels = ['环境温度', '土壤温度', '空气湿度', '土壤湿度',
                   '光照强度', '土壤pH', '氮含量', '叶绿素']

    cm = df_clean[corr_cols].corr()
    cm.index = corr_labels
    cm.columns = corr_labels

    sns.heatmap(cm, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5, square=True, vmin=-1, vmax=1,
                cbar_kws={'label': '相关系数', 'shrink': 0.8})
    ax.set_title('各环境与生理指标相关性矩阵', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig06_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图6 保存完成')


def fig07_health_hist(df_clean: pd.DataFrame) -> None:
    """图7：不同健康状态氮含量与叶绿素分布直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('不同健康状态下氮含量与叶绿素含量分布', fontsize=14, fontweight='bold')

    for status, color in STATUS_PALETTE.items():
        subset = df_clean[df_clean['健康状态'] == status]
        if len(subset) > 0:
            axes[0].hist(subset['Nitrogen_Level'], bins=20, alpha=0.6,
                         color=color, label=status, edgecolor='white')
            axes[1].hist(subset['Chlorophyll_Content'], bins=20, alpha=0.6,
                         color=color, label=status, edgecolor='white')

    axes[0].set_title('氮含量分布（按健康状态）', fontsize=12)
    axes[0].set_xlabel('氮含量 (mg/kg)')
    axes[0].set_ylabel('频次')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title('叶绿素含量分布（按健康状态）', fontsize=12)
    axes[1].set_xlabel('叶绿素含量 (SPAD)')
    axes[1].set_ylabel('频次')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig07_health_hist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图7 保存完成')


def fig09_outlier_detection(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """图9：日均温度异常值自动标注"""
    daily_temp = df_raw.groupby('date')['Ambient_Temperature'].mean().reset_index()
    daily_temp.columns = ['date', 'temp']
    daily_temp['date_idx'] = range(len(daily_temp))

    q1, q3 = daily_temp['temp'].quantile(0.25), daily_temp['temp'].quantile(0.75)
    iqr = q3 - q1
    daily_temp['is_outlier'] = (daily_temp['temp'] < q1 - 1.5 * iqr) | (daily_temp['temp'] > q3 + 1.5 * iqr)
    outlier = daily_temp[daily_temp['is_outlier']]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily_temp['date_idx'], daily_temp['temp'], color='#5D8AA8',
            linewidth=2, alpha=0.85, label='日均温度')
    ax.fill_between(daily_temp['date_idx'], daily_temp['temp'], alpha=0.08, color='#5D8AA8')

    if len(outlier) > 0:
        ax.scatter(outlier['date_idx'], outlier['temp'], color='#E0555A', s=100, zorder=5,
                   label=f'异常值 ({len(outlier)}个)', edgecolors='white', linewidths=1.5)
        for _, row in outlier.iterrows():
            ax.annotate(f'{row["temp"]:.1f}℃', xy=(row['date_idx'], row['temp']),
                        xytext=(row['date_idx'] + 0.5, row['temp'] + 0.3),
                        fontsize=9, color='darkred', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#E0555A', lw=1.2))

    max_idx = daily_temp['temp'].idxmax()
    min_idx = daily_temp['temp'].idxmin()
    ax.annotate(f'最高 {daily_temp.loc[max_idx, "temp"]:.1f}℃',
                xy=(daily_temp.loc[max_idx, 'date_idx'], daily_temp.loc[max_idx, 'temp']),
                xytext=(daily_temp.loc[max_idx, 'date_idx'] - 2, daily_temp.loc[max_idx, 'temp'] + 0.5),
                fontsize=9, color='#C44E52', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#C44E52'))
    ax.annotate(f'最低 {daily_temp.loc[min_idx, "temp"]:.1f}℃',
                xy=(daily_temp.loc[min_idx, 'date_idx'], daily_temp.loc[min_idx, 'temp']),
                xytext=(daily_temp.loc[min_idx, 'date_idx'] + 1, daily_temp.loc[min_idx, 'temp'] - 0.8),
                fontsize=9, color='#4C72B0', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#4C72B0'))

    ax.axhline(q1 - 1.5 * iqr, color='#E5A93D', linestyle='--', alpha=0.6, linewidth=1,
               label=f'IQR下界 ({q1 - 1.5 * iqr:.1f}℃)')
    ax.axhline(q3 + 1.5 * iqr, color='#E5A93D', linestyle='--', alpha=0.6, linewidth=1,
               label=f'IQR上界 ({q3 + 1.5 * iqr:.1f}℃)')

    ax.set_title('日均环境温度时序分析（含异常值自动标注）', fontsize=14, fontweight='bold')
    ax.set_xlabel('监测天数', fontsize=11)
    ax.set_ylabel('环境温度 (℃)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig09_outlier_detection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图9 保存完成')


def fig13_electrochemical(df_clean: pd.DataFrame) -> None:
    """图13：电化学信号与健康状态关联分析"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('电化学信号与植物健康状态关联分析', fontsize=14, fontweight='bold')

    order = ['健康', '中度胁迫', '高度胁迫']
    colors = [STATUS_PALETTE[s] for s in order]
    violin_parts = axes[0].violinplot(
        [df_clean[df_clean['健康状态'] == s]['Electrochemical_Signal'].dropna() for s in order],
        positions=[0, 1, 2], showmeans=True, showmedians=True
    )
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.7)
    axes[0].set_xticks([0, 1, 2]); axes[0].set_xticklabels(order, fontsize=11)
    axes[0].set_ylabel('电化学信号', fontsize=11)
    axes[0].set_title('不同健康状态下电化学信号分布', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)

    for status, color in STATUS_PALETTE.items():
        subset = df_clean[df_clean['健康状态'] == status]
        axes[1].scatter(subset['Soil_Moisture'], subset['Electrochemical_Signal'],
                        c=color, label=status, alpha=0.4, s=20, edgecolors='none')
    axes[1].set_xlabel('土壤湿度 (%)', fontsize=11)
    axes[1].set_ylabel('电化学信号', fontsize=11)
    axes[1].set_title('电化学信号 vs 土壤湿度', fontsize=12)
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig13_electrochemical.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('图13 保存完成')


if __name__ == '__main__':
    from prepare_data import load_data, clean_data
    df_raw = load_data('../data/plant_health_data.csv')
    df_clean = clean_data(df_raw)

    fig00_overview(df_clean)
    fig01_daily_trends(df_clean)
    fig02_plant_comparison(df_clean)
    fig03_health_boxplot(df_clean)
    fig04_light_chlorophyll(df_clean)
    fig05_heatmap(df_clean)
    fig06_correlation(df_clean)
    fig07_health_hist(df_clean)
    fig08_interactive(df_clean)
    fig09_outlier_detection(df_raw, df_clean)
    fig10_anova(df_clean)
    fig11_kmeans(df_clean)
    fig12_regression(df_clean)
    fig13_electrochemical(df_clean)
    print('\n所有图表（含fig00概览 + fig13电化学）生成完成！')
