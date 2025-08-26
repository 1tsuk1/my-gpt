```
"""
STL分解（Seasonal Decomposition Of Time Series By Loess）による
月単位収益データの季節性分析

参考: https://www.salesanalytics.co.jp/datascience/datascience003/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# 日本語表示の設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# グラフスタイルの設定
plt.style.use('ggplot')

class STLSeasonalAnalyzer:
    """STL分解による季節性分析クラス"""
    
    def __init__(self, data, date_col='date', value_col='revenue'):
        """
        Parameters:
        -----------
        data : pd.DataFrame or str
            分析対象のデータフレームまたはCSVファイルパス
        date_col : str
            日付列の名前
        value_col : str
            収益値列の名前
        """
        # データの読み込み
        if isinstance(data, str):
            self.df = pd.read_csv(data, 
                                 index_col=date_col, 
                                 parse_dates=True)
        else:
            self.df = data.copy()
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            self.df = self.df.set_index(date_col)
        
        self.value_col = value_col
        self._prepare_data()
        
    def _prepare_data(self):
        """データの前処理"""
        # 月次データにリサンプリング
        self.df = self.df.resample('M').sum()
        
        # 月と年の情報を追加
        self.df['month'] = self.df.index.month
        self.df['year'] = self.df.index.year
        self.df['month_name'] = self.df.index.strftime('%b')
        
        print("="*60)
        print("データ情報")
        print("="*60)
        print(f"期間: {self.df.index[0].strftime('%Y-%m')} ～ {self.df.index[-1].strftime('%Y-%m')}")
        print(f"データ数: {len(self.df)} ヶ月")
        print(f"平均収益: {self.df[self.value_col].mean():,.0f}")
        print(f"標準偏差: {self.df[self.value_col].std():,.0f}")
        print()
        
    def basic_visualization(self, figsize=(12, 9)):
        """基本的な時系列グラフの描画"""
        plt.rcParams['figure.figsize'] = figsize
        
        # 時系列プロット
        self.df[self.value_col].plot(linewidth=2, color='steelblue')
        plt.title('Monthly Revenue Time Series', fontsize=14, fontweight='bold')
        plt.ylabel('Revenue')
        plt.xlabel('Month')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # データの最初と最後を表示
        print("データサンプル（最初の5ヶ月）:")
        print(self.df[self.value_col].head())
        print("\nデータサンプル（最後の5ヶ月）:")
        print(self.df[self.value_col].tail())
        
    def stl_decomposition(self, period=12, robust=True, seasonal=13):
        """
        STL分解の実行
        
        Parameters:
        -----------
        period : int
            季節周期（月次データの場合は通常12）
        robust : bool
            ロバスト推定を使用するか
        seasonal : int
            季節成分の抽出に使用する窓幅（奇数）
        """
        print("="*60)
        print("STL分解（Seasonal Decomposition Of Time Series By Loess）")
        print("="*60)
        
        # STL分解の実行
        stl = STL(self.df[self.value_col], 
                  period=period, 
                  robust=robust,
                  seasonal=seasonal)
        self.stl_result = stl.fit()
        
        # 分解結果の取得
        self.stl_observed = self.stl_result.observed  # 観測データ
        self.stl_trend = self.stl_result.trend       # トレンド
        self.stl_seasonal = self.stl_result.seasonal # 季節性
        self.stl_resid = self.stl_result.resid      # 残差
        
        # 分解結果の表示
        print("\n分解結果の統計情報:")
        print("-"*40)
        print(f"トレンド成分の範囲: {self.stl_trend.min():.0f} ～ {self.stl_trend.max():.0f}")
        print(f"季節成分の範囲: {self.stl_seasonal.min():.0f} ～ {self.stl_seasonal.max():.0f}")
        print(f"残差の標準偏差: {self.stl_resid.std():.0f}")
        
        # 各成分の寄与度を計算
        total_var = self.stl_observed.var()
        trend_var = self.stl_trend.var()
        seasonal_var = self.stl_seasonal.var()
        resid_var = self.stl_resid.var()
        
        print("\n各成分の寄与度（分散ベース）:")
        print("-"*40)
        print(f"トレンド: {trend_var/total_var*100:.1f}%")
        print(f"季節性: {seasonal_var/total_var*100:.1f}%")
        print(f"残差: {resid_var/total_var*100:.1f}%")
        
        return self.stl_result
    
    def plot_stl_components(self, figsize=(12, 10)):
        """STL分解結果のグラフ化"""
        if not hasattr(self, 'stl_result'):
            print("先にstl_decomposition()を実行してください")
            return
        
        # 標準的なSTL分解プロット
        self.stl_result.plot()
        plt.tight_layout()
        plt.show()
        
        # カスタム分解プロット
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # 1. 元データ
        axes[0].plot(self.df.index, self.stl_observed, 
                    linewidth=1.5, color='darkblue', label='Observed')
        axes[0].set_ylabel('Original')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('STL Decomposition Components', fontsize=14, fontweight='bold')
        
        # 2. トレンド
        axes[1].plot(self.df.index, self.stl_trend, 
                    linewidth=2, color='red', label='Trend')
        axes[1].set_ylabel('Trend')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 季節性
        axes[2].plot(self.df.index, self.stl_seasonal, 
                    linewidth=1, color='green', label='Seasonal')
        axes[2].set_ylabel('Seasonal')
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        # 4. 残差
        axes[3].scatter(self.df.index, self.stl_resid, 
                       s=10, alpha=0.5, color='gray', label='Residual')
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        axes[3].legend(loc='upper left')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_components_overlay(self, figsize=(12, 8)):
        """トレンド、季節性、残差を重ねて表示"""
        if not hasattr(self, 'stl_result'):
            print("先にstl_decomposition()を実行してください")
            return
        
        plt.figure(figsize=figsize)
        
        # すべての成分を同じグラフに描画
        plt.plot(self.df.index, self.stl_trend, 
                linewidth=2, label='Trend', color='red')
        plt.plot(self.df.index, self.stl_seasonal, 
                linewidth=1, label='Seasonal', color='green')
        plt.plot(self.df.index, self.stl_resid, 
                linewidth=0.5, label='Residual', color='gray', alpha=0.5)
        
        plt.title('STL Components Overlay', fontsize=14, fontweight='bold')
        plt.ylabel('Value')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def analyze_seasonal_pattern(self):
        """季節パターンの詳細分析"""
        if not hasattr(self, 'stl_seasonal'):
            print("先にstl_decomposition()を実行してください")
            return
        
        # 月別の季節成分を集計
        seasonal_df = pd.DataFrame({
            'month': self.df['month'],
            'seasonal': self.stl_seasonal
        })
        
        monthly_seasonal = seasonal_df.groupby('month')['seasonal'].agg(['mean', 'std'])
        
        # グラフ作成
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 月別季節成分の平均値
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[0, 0].bar(months, monthly_seasonal['mean'], color='steelblue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Average Seasonal Component by Month', fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Seasonal Effect')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. 季節性の年次変化
        pivot_seasonal = pd.DataFrame({
            'month': self.df['month'],
            'year': self.df['year'],
            'seasonal': self.stl_seasonal
        }).pivot(index='month', columns='year', values='seasonal')
        
        for year in pivot_seasonal.columns:
            axes[0, 1].plot(pivot_seasonal.index, pivot_seasonal[year], 
                          marker='o', label=f'{year}', alpha=0.7)
        axes[0, 1].set_title('Seasonal Pattern by Year', fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Seasonal Effect')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(range(1, 13))
        
        # 3. 季節性指数（基準値100）
        base_value = self.df[self.value_col].mean()
        seasonal_index = ((monthly_seasonal['mean'] + base_value) / base_value) * 100
        
        axes[1, 0].plot(months, seasonal_index, marker='o', 
                       linewidth=2, markersize=8, color='darkgreen')
        axes[1, 0].axhline(y=100, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Seasonal Index (Base=100)', fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Index')
        axes[1, 0].grid(True, alpha=0.3)
        
        # インデックス値を表示
        for i, (month, value) in enumerate(zip(months, seasonal_index)):
            axes[1, 0].text(i, value + 2, f'{value:.1f}', 
                          ha='center', fontsize=8)
        
        # 4. ボックスプロット
        seasonal_by_month = [seasonal_df[seasonal_df['month'] == m]['seasonal'].values 
                           for m in range(1, 13)]
        bp = axes[1, 1].boxplot(seasonal_by_month, labels=months, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[1, 1].set_title('Seasonal Component Distribution by Month', fontweight='bold')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Seasonal Effect')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # 統計サマリーの出力
        print("\n"+"="*60)
        print("季節パターン分析結果")
        print("="*60)
        print("\n月別季節性指数（年平均=100）:")
        print("-"*40)
        for month, idx in zip(months, seasonal_index):
            bar_length = int((idx - 80) / 2)  # 視覚的なバー表示
            if bar_length > 0:
                bar = "+" * bar_length
            else:
                bar = "-" * abs(bar_length)
            print(f"{month}: {idx:6.1f} {bar}")
        
        # 最大・最小月の特定
        max_month = months[seasonal_index.argmax()]
        min_month = months[seasonal_index.argmin()]
        print(f"\n最高季節性月: {max_month} (Index: {seasonal_index.max():.1f})")
        print(f"最低季節性月: {min_month} (Index: {seasonal_index.min():.1f})")
        print(f"季節変動幅: {seasonal_index.max() - seasonal_index.min():.1f}ポイント")
        
        return seasonal_index
    
    def trend_analysis(self, figsize=(14, 6)):
        """トレンド成分の詳細分析"""
        if not hasattr(self, 'stl_trend'):
            print("先にstl_decomposition()を実行してください")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. トレンドと元データの比較
        axes[0].plot(self.df.index, self.stl_observed, 
                    linewidth=1, alpha=0.5, label='Original', color='gray')
        axes[0].plot(self.df.index, self.stl_trend, 
                    linewidth=2, label='Trend', color='red')
        axes[0].set_title('Original vs Trend', fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Revenue')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. トレンドの変化率
        trend_pct_change = self.stl_trend.pct_change() * 100
        axes[1].plot(self.df.index, trend_pct_change, 
                    linewidth=1, color='darkblue')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_title('Trend Growth Rate (%)', fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Growth Rate (%)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # トレンド統計
        print("\n"+"="*60)
        print("トレンド分析結果")
        print("="*60)
        print(f"期間全体の成長率: {(self.stl_trend.iloc[-1] / self.stl_trend.iloc[0] - 1) * 100:.1f}%")
        print(f"平均月次成長率: {trend_pct_change.mean():.2f}%")
        print(f"成長率の標準偏差: {trend_pct_change.std():.2f}%")
        
    def residual_diagnostics(self, lags=24):
        """残差の診断"""
        if not hasattr(self, 'stl_resid'):
            print("先にstl_decomposition()を実行してください")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 残差の時系列プロット
        axes[0, 0].scatter(self.df.index, self.stl_resid, 
                         s=10, alpha=0.5, color='gray')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Residuals over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差のヒストグラム
        axes[0, 1].hist(self.stl_resid.dropna(), bins=30, 
                       edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].set_title('Residual Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. ACF
        plot_acf(self.stl_resid.dropna(), lags=lags, ax=axes[1, 0])
        axes[1, 0].set_title('Residual ACF', fontweight='bold')
        
        # 4. Q-Qプロット
        from scipy import stats
        stats.probplot(self.stl_resid.dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Ljung-Box検定
        lb_test = acorr_ljungbox(self.stl_resid.dropna(), 
                                lags=12, return_df=True)
        
        print("\n"+"="*60)
        print("残差診断結果")
        print("="*60)
        print(f"残差の平均: {self.stl_resid.mean():.2f}")
        print(f"残差の標準偏差: {self.stl_resid.std():.2f}")
        print(f"残差の歪度: {self.stl_resid.skew():.2f}")
        print(f"残差の尖度: {self.stl_resid.kurtosis():.2f}")
        
        print("\nLjung-Box検定（残差の自己相関）:")
        print("-"*40)
        print(lb_test[['lb_stat', 'lb_pvalue']].tail(1))
        
        if lb_test['lb_pvalue'].iloc[-1] > 0.05:
            print("\n結果: 残差に有意な自己相関なし（モデル適合良好）")
        else:
            print("\n結果: 残差に有意な自己相関あり（モデル改善の余地あり）")
    
    def compare_decomposition_methods(self):
        """STL分解と古典的分解の比較"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # STL分解（既に実行済み）
        axes[0, 0].plot(self.stl_trend, linewidth=2, color='red')
        axes[0, 0].set_title('STL - Trend', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.stl_seasonal, linewidth=1, color='green')
        axes[1, 0].set_title('STL - Seasonal', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[2, 0].plot(self.stl_resid, linewidth=0.5, color='gray')
        axes[2, 0].set_title('STL - Residual', fontweight='bold')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 古典的分解（加法モデル）
        classical = seasonal_decompose(self.df[self.value_col], 
                                      model='additive', 
                                      period=12)
        
        axes[0, 1].plot(classical.trend, linewidth=2, color='red')
        axes[0, 1].set_title('Classical - Trend', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 1].plot(classical.seasonal, linewidth=1, color='green')
        axes[1, 1].set_title('Classical - Seasonal', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[2, 1].plot(classical.resid, linewidth=0.5, color='gray')
        axes[2, 1].set_title('Classical - Residual', fontweight='bold')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n"+"="*60)
        print("STL分解 vs 古典的分解の比較")
        print("="*60)
        print("\nSTL分解の特徴:")
        print("- ロバスト推定により外れ値の影響を軽減")
        print("- 季節パターンの時間変化を許容")
        print("- より柔軟なトレンド抽出")
        print("\n古典的分解の特徴:")
        print("- 移動平均ベースのシンプルな手法")
        print("- 季節パターンは固定")
        print("- 計算が高速")

# サンプルデータ生成関数
def generate_sample_data():
    """サンプル収益データの生成（記事の例を参考に）"""
    np.random.seed(42)
    
    # 4年分の月次データ
    dates = pd.date_range(start='2021-01-01', end='2024-12-31', freq='M')
    
    # トレンド成分（上昇傾向）
    trend = np.linspace(100000, 200000, len(dates))
    
    # 季節成分（12ヶ月周期）
    seasonal_pattern = [
        0.90, 0.85, 0.95, 1.00, 1.05, 1.10,  # 1-6月
        1.15, 1.12, 1.08, 1.05, 1.20, 1.35   # 7-12月（12月が最高）
    ]
    seasonal = np.tile(seasonal_pattern, len(dates)//12 + 1)[:len(dates)]
    
    # ノイズ
    noise = np.random.normal(0, 5000, len(dates))
    
    # 合成（加法モデル）
    revenue = trend + (trend * (seasonal - 1)) + noise
    
    df = pd.DataFrame({
        'date': dates,
        'revenue': revenue
    })
    
    return df

# メイン実行コード
if __name__ == "__main__":
    print("="*60)
    print("STL分解による月単位収益データの季節性分析")
    print("="*60)
    print()
    
    # データの準備
    print("サンプルデータを生成中...")
    df = generate_sample_data()
    
    # 分析クラスのインスタンス化
    analyzer = STLSeasonalAnalyzer(df, date_col='date', value_col='revenue')
    
    # 1. 基本的な可視化
    print("\n【1. 基本的な時系列グラフ】")
    analyzer.basic_visualization()
    
    # 2. STL分解の実行
    print("\n【2. STL分解の実行】")
    stl_result = analyzer.stl_decomposition(period=12, robust=True)
    
    # 3. STL分解結果の可視化
    print("\n【3. STL分解結果の可視化】")
    analyzer.plot_stl_components()
    
    # 4. 成分の重ね合わせ表示
    print("\n【4. トレンド・季節性・残差の重ね合わせ】")
    analyzer.plot_components_overlay()
    
    # 5. 季節パターンの詳細分析
    print("\n【5. 季節パターンの詳細分析】")
    seasonal_index = analyzer.analyze_seasonal_pattern()
    
    # 6. トレンド分析
    print("\n【6. トレンド分析】")
    analyzer.trend_analysis()
    
    # 7. 残差診断
    print("\n【7. 残差の診断】")
    analyzer.residual_diagnostics()
    
    # 8. 分解手法の比較
    print("\n【8. STL分解と古典的分解の比較】")
    analyzer.compare_decomposition_methods()
    
    print("\n"+"="*60)
    print("分析完了")
    print("="*60)
```
