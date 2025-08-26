```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# 日本語表示の設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SeasonalAnalysis:
    """月単位収益データの季節性分析クラス"""
    
    def __init__(self, data, date_col='date', value_col='revenue'):
        """
        Parameters:
        -----------
        data : pd.DataFrame
            分析対象のデータフレーム
        date_col : str
            日付列の名前
        value_col : str
            収益値列の名前
        """
        self.df = data.copy()
        self.date_col = date_col
        self.value_col = value_col
        self._prepare_data()
        
    def _prepare_data(self):
        """データの前処理"""
        # 日付をインデックスに設定
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df.set_index(self.date_col)
        
        # 月次データにリサンプリング（必要に応じて）
        self.df = self.df.resample('M').sum()
        
        # 月と年の列を追加
        self.df['month'] = self.df.index.month
        self.df['year'] = self.df.index.year
        self.df['month_name'] = self.df.index.strftime('%b')
        
    def plot_time_series(self, figsize=(14, 6)):
        """時系列プロットの作成"""
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # 1. 時系列プロット
        axes[0].plot(self.df.index, self.df[self.value_col], linewidth=2)
        axes[0].set_title('Revenue Time Series', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Revenue')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 年別の月次推移
        pivot_data = self.df.pivot_table(
            values=self.value_col, 
            index='month', 
            columns='year'
        )
        
        for year in pivot_data.columns:
            axes[1].plot(pivot_data.index, pivot_data[year], 
                        marker='o', label=f'{year}')
        
        axes[1].set_title('Monthly Revenue by Year', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Revenue')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(range(1, 13))
        
        plt.tight_layout()
        plt.show()
        
    def seasonal_decomposition(self, model='additive', period=12):
        """
        季節性分解
        
        Parameters:
        -----------
        model : str
            'additive' (加法モデル) or 'multiplicative' (乗法モデル)
        period : int
            季節周期（月次データの場合は通常12）
        """
        # 分解の実行
        decomposition = seasonal_decompose(
            self.df[self.value_col], 
            model=model, 
            period=period
        )
        
        # プロット
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        # 原系列
        self.df[self.value_col].plot(ax=axes[0])
        axes[0].set_title('Original Data', fontweight='bold')
        axes[0].set_ylabel('Revenue')
        
        # トレンド
        decomposition.trend.plot(ax=axes[1])
        axes[1].set_title('Trend Component', fontweight='bold')
        axes[1].set_ylabel('Trend')
        
        # 季節成分
        decomposition.seasonal.plot(ax=axes[2])
        axes[2].set_title('Seasonal Component', fontweight='bold')
        axes[2].set_ylabel('Seasonal')
        
        # 残差
        decomposition.resid.plot(ax=axes[3])
        axes[3].set_title('Residual Component', fontweight='bold')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        
        plt.tight_layout()
        plt.show()
        
        return decomposition
    
    def analyze_seasonal_patterns(self):
        """月別の季節パターン分析"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 月別の箱ひげ図
        monthly_data = []
        months = []
        for month in range(1, 13):
            month_data = self.df[self.df['month'] == month][self.value_col]
            monthly_data.append(month_data)
            months.append(pd.to_datetime(f'2024-{month:02d}-01').strftime('%b'))
        
        axes[0, 0].boxplot(monthly_data, labels=months)
        axes[0, 0].set_title('Monthly Revenue Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Revenue')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 月別平均収益
        monthly_avg = self.df.groupby('month')[self.value_col].mean()
        axes[0, 1].bar(months, monthly_avg.values, color='steelblue')
        axes[0, 1].set_title('Average Monthly Revenue', fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average Revenue')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. 季節性指数
        overall_mean = self.df[self.value_col].mean()
        seasonal_index = (monthly_avg / overall_mean) * 100
        
        axes[1, 0].plot(months, seasonal_index.values, marker='o', linewidth=2)
        axes[1, 0].axhline(y=100, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Seasonal Index', fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Index (Average = 100)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. ヒートマップ
        pivot_table = self.df.pivot_table(
            values=self.value_col,
            index='month',
            columns='year',
            aggfunc='sum'
        )
        
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd', 
                   ax=axes[1, 1], cbar_kws={'label': 'Revenue'})
        axes[1, 1].set_title('Revenue Heatmap by Month and Year', fontweight='bold')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Month')
        
        plt.tight_layout()
        plt.show()
        
        # 統計情報の出力
        print("=" * 50)
        print("Seasonal Analysis Summary")
        print("=" * 50)
        print("\n季節性指数（月別）:")
        for i, month in enumerate(months):
            print(f"{month}: {seasonal_index.iloc[i]:.1f}")
        
        # 最大・最小月の特定
        max_month = months[seasonal_index.argmax()]
        min_month = months[seasonal_index.argmin()]
        print(f"\n最高収益月: {max_month} (Index: {seasonal_index.max():.1f})")
        print(f"最低収益月: {min_month} (Index: {seasonal_index.min():.1f})")
        
        return seasonal_index
    
    def moving_average_analysis(self, windows=[3, 6, 12]):
        """移動平均分析"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 原系列
        ax.plot(self.df.index, self.df[self.value_col], 
                label='Original', alpha=0.7, linewidth=1)
        
        # 移動平均
        for window in windows:
            ma = self.df[self.value_col].rolling(window=window).mean()
            ax.plot(self.df.index, ma, 
                   label=f'{window}-Month MA', linewidth=2)
        
        ax.set_title('Moving Average Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Revenue')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def autocorrelation_analysis(self, lags=24):
        """自己相関分析"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # ACF
        plot_acf(self.df[self.value_col].dropna(), lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)', fontweight='bold')
        
        # PACF
        plot_pacf(self.df[self.value_col].dropna(), lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Ljung-Box検定（季節性の統計的検定）
        lb_test = acorr_ljungbox(self.df[self.value_col].dropna(), 
                                 lags=12, return_df=True)
        
        print("\n" + "=" * 50)
        print("Ljung-Box Test for Seasonality")
        print("=" * 50)
        print(lb_test)
        
        if lb_test['lb_pvalue'].iloc[11] < 0.05:
            print("\n結果: 有意な季節性が検出されました (p < 0.05)")
        else:
            print("\n結果: 有意な季節性は検出されませんでした (p >= 0.05)")
    
    def calculate_seasonal_strength(self, decomposition=None):
        """季節性の強度を計算"""
        if decomposition is None:
            decomposition = seasonal_decompose(
                self.df[self.value_col], 
                model='additive', 
                period=12
            )
        
        # 季節性の強度を計算
        seasonal_var = decomposition.seasonal.var()
        residual_var = decomposition.resid.var()
        
        # 季節性の強度（0から1の範囲）
        seasonal_strength = 1 - (residual_var / (seasonal_var + residual_var))
        
        print("\n" + "=" * 50)
        print("Seasonal Strength Analysis")
        print("=" * 50)
        print(f"季節性の強度: {seasonal_strength:.3f}")
        
        if seasonal_strength > 0.64:
            print("評価: 強い季節性")
        elif seasonal_strength > 0.4:
            print("評価: 中程度の季節性")
        else:
            print("評価: 弱い季節性")
        
        return seasonal_strength

# 使用例
def generate_sample_data():
    """サンプルデータの生成"""
    np.random.seed(42)
    
    # 3年分の月次データを生成
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='M')
    
    # ベースライン + トレンド + 季節性 + ノイズ
    baseline = 100000
    trend = np.linspace(0, 50000, len(dates))
    
    # 季節パターン（12月が最高、2月が最低）
    seasonal_pattern = [0.85, 0.80, 0.90, 0.95, 1.00, 1.05, 
                       1.10, 1.15, 1.10, 1.05, 1.15, 1.30]
    seasonal = np.tile(seasonal_pattern, len(dates)//12 + 1)[:len(dates)]
    
    noise = np.random.normal(0, 5000, len(dates))
    
    revenue = baseline + trend + (baseline * (seasonal - 1)) + noise
    
    df = pd.DataFrame({
        'date': dates,
        'revenue': revenue
    })
    
    return df

# メイン実行コード
if __name__ == "__main__":
    # サンプルデータの生成（実際のデータがある場合は読み込み）
    print("Generating sample data...")
    df = generate_sample_data()
    
    # 分析クラスのインスタンス化
    analyzer = SeasonalAnalysis(df, date_col='date', value_col='revenue')
    
    print("\n1. Time Series Visualization")
    analyzer.plot_time_series()
    
    print("\n2. Seasonal Decomposition")
    decomposition = analyzer.seasonal_decomposition(model='additive')
    
    print("\n3. Seasonal Pattern Analysis")
    seasonal_index = analyzer.analyze_seasonal_patterns()
    
    print("\n4. Moving Average Analysis")
    analyzer.moving_average_analysis()
    
    print("\n5. Autocorrelation Analysis")
    analyzer.autocorrelation_analysis()
    
    print("\n6. Seasonal Strength Calculation")
    strength = analyzer.calculate_seasonal_strength(decomposition)
```
