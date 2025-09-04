```
"""
STL分解（Seasonal Decomposition Of Time Series By Loess）による
月単位収益データの季節性分析 - 加法/乗法モデル対応版

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
    """STL分解による季節性分析クラス（加法/乗法モデル対応）"""
    
    def __init__(self, data, date_col='date', value_col='revenue', model='additive'):
        """
        Parameters:
        -----------
        data : pd.DataFrame or str
            分析対象のデータフレームまたはCSVファイルパス
        date_col : str
            日付列の名前
        value_col : str
            収益値列の名前
        model : str
            'additive' (加法モデル) または 'multiplicative' (乗法モデル)
        """
        # モデルタイプの検証
        if model not in ['additive', 'multiplicative']:
            raise ValueError("model must be 'additive' or 'multiplicative'")
        
        self.model = model
        
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
        
        # 乗法モデルの場合、データが正の値であることを確認
        if self.model == 'multiplicative':
            if (self.df[self.value_col] <= 0).any():
                print("警告: 乗法モデルには正の値が必要です。")
                # ゼロや負の値がある場合、最小値を正にシフト
                min_val = self.df[self.value_col].min()
                if min_val <= 0:
                    self.shift_value = abs(min_val) + 1
                    self.df[self.value_col] = self.df[self.value_col] + self.shift_value
                    print(f"データを {self.shift_value} だけシフトしました。")
                else:
                    self.shift_value = 0
            else:
                self.shift_value = 0
        
        # 月と年の情報を追加
        self.df['month'] = self.df.index.month
        self.df['year'] = self.df.index.year
        self.df['month_name'] = self.df.index.strftime('%b')
        
        print("="*60)
        print(f"データ情報（モデル: {self.model.upper()}）")
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
        plt.title(f'Monthly Revenue Time Series ({self.model.capitalize()} Model)', 
                 fontsize=14, fontweight='bold')
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
        STL分解の実行（乗法モデル対応）
        
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
        print(f"STL分解（{self.model.upper()} MODEL）")
        print("="*60)
        
        if self.model == 'multiplicative':
            # 乗法モデルの場合、対数変換してSTLを適用
            log_data = np.log(self.df[self.value_col])
            stl = STL(log_data, period=period, robust=robust, seasonal=seasonal)
            stl_result_log = stl.fit()
            
            # 元のスケールに戻す
            self.stl_observed = self.df[self.value_col]
            self.stl_trend = np.exp(stl_result_log.trend)
            self.stl_seasonal = np.exp(stl_result_log.seasonal)
            self.stl_resid = np.exp(stl_result_log.resid)
            
            # 内部的にログスケールの結果も保持
            self.stl_result_log = stl_result_log
            self.stl_result = stl_result_log  # 互換性のため
            
        else:
            # 加法モデルの場合、通常のSTL分解
            stl = STL(self.df[self.value_col], 
                      period=period, 
                      robust=robust,
                      seasonal=seasonal)
            self.stl_result = stl.fit()
            
            # 分解結果の取得
            self.stl_observed = self.stl_result.observed
            self.stl_trend = self.stl_result.trend
            self.stl_seasonal = self.stl_result.seasonal
            self.stl_resid = self.stl_result.resid
        
        # 分解結果の表示
        print("\n分解結果の統計情報:")
        print("-"*40)
        
        if self.model == 'multiplicative':
            print(f"トレンド成分の範囲: {self.stl_trend.min():.0f} ～ {self.stl_trend.max():.0f}")
            print(f"季節成分の範囲（乗数）: {self.stl_seasonal.min():.3f} ～ {self.stl_seasonal.max():.3f}")
            print(f"残差の範囲（乗数）: {self.stl_resid.min():.3f} ～ {self.stl_resid.max():.3f}")
            
            # 検証：元のデータ ≈ trend × seasonal × residual
            reconstructed = self.stl_trend * self.stl_seasonal * self.stl_resid
            reconstruction_error = np.mean(np.abs(self.stl_observed - reconstructed))
            print(f"\n再構成誤差（平均絶対誤差）: {reconstruction_error:.2f}")
            
        else:
            print(f"トレンド成分の範囲: {self.stl_trend.min():.0f} ～ {self.stl_trend.max():.0f}")
            print(f"季節成分の範囲: {self.stl_seasonal.min():.0f} ～ {self.stl_seasonal.max():.0f}")
            print(f"残差の標準偏差: {self.stl_resid.std():.0f}")
            
            # 各成分の寄与度を計算（加法モデルのみ）
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
        """STL分解結果のグラフ化（乗法モデル対応）"""
        if not hasattr(self, 'stl_result'):
            print("先にstl_decomposition()を実行してください")
            return
        
        # カスタム分解プロット
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # タイトル
        model_label = "Multiplicative" if self.model == 'multiplicative' else "Additive"
        
        # 1. 元データ
        axes[0].plot(self.df.index, self.stl_observed, 
                    linewidth=1.5, color='darkblue', label='Observed')
        axes[0].set_ylabel('Original')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f'STL Decomposition Components ({model_label} Model)', 
                         fontsize=14, fontweight='bold')
        
        # 2. トレンド
        axes[1].plot(self.df.index, self.stl_trend, 
                    linewidth=2, color='red', label='Trend')
        axes[1].set_ylabel('Trend')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 季節性
        axes[2].plot(self.df.index, self.stl_seasonal, 
                    linewidth=1, color='green', label='Seasonal')
        if self.model == 'multiplicative':
            axes[2].set_ylabel('Seasonal\n(Multiplier)')
            axes[2].axhline(y=1, color='black', linestyle='--', alpha=0.3)
        else:
            axes[2].set_ylabel('Seasonal')
            axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        # 4. 残差
        axes[3].scatter(self.df.index, self.stl_resid, 
                       s=10, alpha=0.5, color='gray', label='Residual')
        if self.model == 'multiplicative':
            axes[3].axhline(y=1, color='black', linestyle='--', alpha=0.3)
            axes[3].set_ylabel('Residual\n(Multiplier)')
        else:
            axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        axes[3].legend(loc='upper left')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def analyze_seasonal_pattern(self):
        """季節パターンの詳細分析（乗法モデル対応）"""
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
        
        if self.model == 'multiplicative':
            # 乗法モデルの場合、1を基準線として表示
            bars = axes[0, 0].bar(months, monthly_seasonal['mean'], color='steelblue')
            axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5)
            axes[0, 0].set_ylabel('Seasonal Effect (Multiplier)')
            
            # バーの上に値を表示
            for bar, value in zip(bars, monthly_seasonal['mean']):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            axes[0, 0].bar(months, monthly_seasonal['mean'], color='steelblue')
            axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 0].set_ylabel('Seasonal Effect')
            
        axes[0, 0].set_title('Average Seasonal Component by Month', fontweight='bold')
        axes[0, 0].set_xlabel('Month')
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
        
        if self.model == 'multiplicative':
            axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5)
            
        axes[0, 1].set_title('Seasonal Pattern by Year', fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Seasonal Effect')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(range(1, 13))
        
        # 3. 季節性指数
        if self.model == 'multiplicative':
            # 乗法モデルの場合、そのまま季節成分を指数として使用（×100）
            seasonal_index = monthly_seasonal['mean'] * 100
        else:
            # 加法モデルの場合、基準値100の指数を計算
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
        
        if self.model == 'multiplicative':
            axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5)
        else:
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
        axes[1, 1].set_title('Seasonal Component Distribution by Month', fontweight='bold')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Seasonal Effect')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # 統計サマリーの出力
        print("\n"+"="*60)
        print(f"季節パターン分析結果（{self.model.upper()} MODEL）")
        print("="*60)
        print("\n月別季節性指数（年平均=100）:")
        print("-"*40)
        for month, idx in zip(months, seasonal_index):
            bar_length = int((idx - 80) / 2)
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
    
    def compare_models(self):
        """加法モデルと乗法モデルの比較（両方のSTLとseasonal_decompose）"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        
        # データを取得
        data = self.df[self.value_col]
        
        # 1. STL - 加法モデル
        stl_add = STL(data, period=12).fit()
        axes[0, 0].plot(stl_add.trend, linewidth=2, color='red')
        axes[0, 0].set_title('STL Additive - Trend', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[1, 0].plot(stl_add.seasonal, linewidth=1, color='green')
        axes[1, 0].set_title('STL Additive - Seasonal', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        axes[2, 0].plot(stl_add.resid, linewidth=0.5, color='gray')
        axes[2, 0].set_title('STL Additive - Residual', fontweight='bold')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # 2. STL - 乗法モデル（対数変換）
        log_data = np.log(data)
        stl_mult_log = STL(log_data, period=12).fit()
        stl_mult_trend = np.exp(stl_mult_log.trend)
        stl_mult_seasonal = np.exp(stl_mult_log.seasonal)
        stl_mult_resid = np.exp(stl_mult_log.resid)
        
        axes[0, 1].plot(stl_mult_trend, linewidth=2, color='red')
        axes[0, 1].set_title('STL Multiplicative - Trend', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 1].plot(stl_mult_seasonal, linewidth=1, color='green')
        axes[1, 1].set_title('STL Multiplicative - Seasonal', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=1, color='black', linestyle='--', alpha=0.3)
        
        axes[2, 1].plot(stl_mult_resid, linewidth=0.5, color='gray')
        axes[2, 1].set_title('STL Multiplicative - Residual', fontweight='bold')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].axhline(y=1, color='black', linestyle='--', alpha=0.3)
        
        # 3. Classical - 加法モデル
        classical_add = seasonal_decompose(data, model='additive', period=12)
        axes[0, 2].plot(classical_add.trend, linewidth=2, color='red')
        axes[0, 2].set_title('Classical Additive - Trend', fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 2].plot(classical_add.seasonal, linewidth=1, color='green')
        axes[1, 2].set_title('Classical Additive - Seasonal', fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        axes[2, 2].plot(classical_add.resid, linewidth=0.5, color='gray')
        axes[2, 2].set_title('Classical Additive - Residual', fontweight='bold')
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # 4. Classical - 乗法モデル
        classical_mult = seasonal_decompose(data, model='multiplicative', period=12)
        axes[0, 3].plot(classical_mult.trend, linewidth=2, color='red')
        axes[0, 3].set_title('Classical Multiplicative - Trend', fontweight='bold')
        axes[0, 3].grid(True, alpha=0.3)
        
        axes[1, 3].plot(classical_mult.seasonal, linewidth=1, color='green')
        axes[1, 3].set_title('Classical Multiplicative - Seasonal', fontweight='bold')
        axes[1, 3].grid(True, alpha=0.3)
        axes[1, 3].axhline(y=1, color='black', linestyle='--', alpha=0.3)
        
        axes[2, 3].plot(classical_mult.resid, linewidth=0.5, color='gray')
        axes[2, 3].set_title('Classical Multiplicative - Residual', fontweight='bold')
        axes[2, 3].grid(True, alpha=0.3)
        axes[2, 3].axhline(y=1, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n"+"="*60)
        print("4つの分解手法の比較")
        print("="*60)
        print("\n1. STL - 加法モデル:")
        print("   - ロバスト推定、柔軟なトレンド")
        print("   - Y = Trend + Seasonal + Residual")
        
        print("\n2. STL - 乗法モデル（対数変換）:")
        print("   - 対数変換によりSTLで乗法モデルを実現")
        print("   - Y = Trend × Seasonal × Residual")
        
        print("\n3. Classical - 加法モデル:")
        print("   - 単純な移動平均ベース")
        print("   - Y = Trend + Seasonal + Residual")
        
        print("\n4. Classical - 乗法モデル:")
        print("   - ネイティブな乗法モデルサポート")
        print("   - Y = Trend × Seasonal × Residual")

# サンプルデータ生成関数（乗法モデル用も追加）
def generate_sample_data(model_type='multiplicative'):
    """サンプル収益データの生成"""
    np.random.seed(42)
    
    # 4年分の月次データ
    dates = pd.date_range(start='2021-01-01', end='2024-12-31', freq='M')
    
    if model_type == 'multiplicative':
        # 乗法モデル用データ
        # トレンド成分（上昇傾向）
        trend = np.linspace(100000, 200000, len(dates))
        
        # 季節成分（乗数として）
        seasonal_pattern = [
            0.90, 0.85, 0.95, 1.00, 1.05, 1.10,  # 1-6月
            1.15, 1.12, 1.08, 1.05, 1.20, 1.35   # 7-12月（12月が最高）
        ]
        seasonal = np.tile(seasonal_pattern, len(dates)//12 + 1)[:len(dates)]
        
        # ノイズ（乗数として、平均1の正規分布）
        noise = np.random.normal(1, 0.05, len(dates))
        
        # 合成（乗法モデル）
        revenue = trend * seasonal * noise
        
    else:
        # 加法モデル用データ
        trend = np.linspace(100000, 200000, len(dates))
        seasonal_pattern = np.array([
            -10000, -15000, -5000, 0, 5000, 10000,  # 1-6月
            15000, 12000, 8000, 5000, 20000, 35000   # 7-12月
        ])
        seasonal = np.tile(seasonal_pattern, len(dates)//12 + 1)[:len(dates)]
        noise = np.random.normal(0, 5000, len(dates))
        
        # 合成（加法モデル）
        revenue = trend + seasonal + noise
    
    df = pd.DataFrame({
        'date': dates,
        'revenue': revenue
    })
    
    return df

# メイン実行コード
if __name__ == "__main__":
    print("="*60)
    print("STL分解による月単位収益データの季節性分析")
    print("加法モデル・乗法モデル対応版")
    print("="*60)
    print()
    
    # 乗法モデルでの分析を実行
    print("【乗法モデルでの分析】")
    print("="*60)
    
    # データの準備（乗法モデル用）
    print("サンプルデータを生成中（乗法モデル用）...")
    df_mult = generate_sample_data(model_type='multiplicative')
    
    # 分析クラスのインスタンス化（乗法モデル）
    analyzer_mult = STLSeasonalAnalyzer(df_mult, date_col='date', 
                                        value_col='revenue', 
                                        model='multiplicative')
    
    # 1. 基本的な可視化
    print("\n【1. 基本的な時系列グラフ】")
    analyzer_mult.basic_visualization()
    
    # 2. STL分解の実行
    print("\n【2. STL分解の実行（乗法モデル）】")
    stl_result_mult = analyzer_mult.stl_decomposition(period=12, robust=True)
    
    # 3. STL分解結果の可視化
    print("\n【3. STL分解結果の可視化】")
    analyzer_mult.plot_stl_components()
    
    # 4. 季節パターンの詳細分析
    print("\n【4. 季節パターンの詳細分析】")
    seasonal_index_mult = analyzer_mult.analyze_seasonal_pattern()
    
    # 5. 4つのモデルの比較
    print("\n【5. 加法・乗法モデルの比較】")
    analyzer_mult.compare_models()
    
    print("\n"+"="*60)
    print("\n【加法モデルとの比較のため、同じデータで加法モデルも実行】")
    print("="*60)
    
    # 同じデータで加法モデルも試す
    analyzer_add = STLSeasonalAnalyzer(df_mult, date_col='date', 
                                       value_col='revenue', 
                                       model='additive')
    
    print("\n【加法モデルでのSTL分解】")
    stl_result_add = analyzer_add.stl_decomposition(period=12, robust=True)
    analyzer_add.plot_stl_components()
    
    print("\n"+"="*60)
    print("分析完了")
    print("="*60)
    print("\n乗法モデルの特徴:")
    print("- 季節変動が収益レベルに比例する場合に適している")
    print("- 季節成分とトレンドが相互作用を持つ")
    print("- 成長率が一定の時系列データに適合しやすい")
    print("\n加法モデルの特徴:")
    print("- 季節変動が一定の場合に適している")
    print("- 各成分が独立している")
    print("- 解釈が直感的で扱いやすい")
```
