```
"""
seasonal_decomposeによる月単位収益データの季節性分析
加法モデル・乗法モデル完全対応版

参考: https://www.salesanalytics.co.jp/datascience/datascience003/
"""

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

# グラフスタイルの設定
plt.style.use('ggplot')

class SeasonalAnalyzer:
    """seasonal_decomposeによる季節性分析クラス（加法/乗法モデル対応）"""
    
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
        
    def basic_visualization(self, figsize=(12, 6)):
        """基本的な時系列グラフの描画"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 時系列プロット
        axes[0].plot(self.df.index, self.df[self.value_col], 
                    linewidth=2, color='steelblue')
        axes[0].set_title(f'Monthly Revenue Time Series', fontweight='bold')
        axes[0].set_ylabel('Revenue')
        axes[0].set_xlabel('Month')
        axes[0].grid(True, alpha=0.3)
        
        # 月別箱ひげ図
        monthly_data = pd.DataFrame({
            'month': self.df.index.month,
            'revenue': self.df[self.value_col].values
        })
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        monthly_boxplot_data = [monthly_data[monthly_data['month'] == m+1]['revenue'].values 
                                for m in range(12)]
        
        bp = axes[1].boxplot(monthly_boxplot_data, labels=months, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[1].set_title('Revenue Distribution by Month', fontweight='bold')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Revenue')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # データサンプル表示
        print("データサンプル（最初の5ヶ月）:")
        print(self.df[self.value_col].head())
        print("\nデータサンプル（最後の5ヶ月）:")
        print(self.df[self.value_col].tail())
        
    def seasonal_decomposition(self, period=12, extrapolate_trend='freq'):
        """
        seasonal_decompose による時系列分解
        
        Parameters:
        -----------
        period : int
            季節周期（月次データの場合は通常12）
        extrapolate_trend : str or int
            トレンドの外挿方法
        """
        print("="*60)
        print(f"Seasonal Decomposition（{self.model.upper()} MODEL）")
        print("="*60)
        
        # seasonal_decompose の実行
        self.decomposition = seasonal_decompose(
            self.df[self.value_col],
            model=self.model,  # 'additive' or 'multiplicative'
            period=period,
            extrapolate_trend=extrapolate_trend
        )
        
        # 分解結果の取得
        self.observed = self.decomposition.observed
        self.trend = self.decomposition.trend
        self.seasonal = self.decomposition.seasonal
        self.resid = self.decomposition.resid
        
        # 分解結果の統計情報を表示
        print("\n分解結果の統計情報:")
        print("-"*40)
        
        if self.model == 'multiplicative':
            print(f"トレンド成分の範囲: {self.trend.min():,.0f} ～ {self.trend.max():,.0f}")
            print(f"季節成分の範囲（乗数）: {self.seasonal.min():.3f} ～ {self.seasonal.max():.3f}")
            print(f"季節成分の平均: {self.seasonal.mean():.3f}")
            print(f"残差の範囲（乗数）: {self.resid.dropna().min():.3f} ～ {self.resid.dropna().max():.3f}")
            print(f"残差の平均: {self.resid.dropna().mean():.3f}")
            
            # 再構成検証
            reconstructed = self.trend * self.seasonal * self.resid
            reconstruction_error = np.nanmean(np.abs(self.observed - reconstructed))
            print(f"\n再構成誤差（平均絶対誤差）: {reconstruction_error:.2f}")
            
        else:
            print(f"トレンド成分の範囲: {self.trend.min():,.0f} ～ {self.trend.max():,.0f}")
            print(f"季節成分の範囲: {self.seasonal.min():,.0f} ～ {self.seasonal.max():,.0f}")
            print(f"残差の標準偏差: {self.resid.dropna().std():,.0f}")
            
            # 各成分の寄与度を計算
            total_var = self.observed.var()
            trend_var = self.trend.dropna().var()
            seasonal_var = self.seasonal.var()
            resid_var = self.resid.dropna().var()
            
            print("\n各成分の寄与度（分散ベース）:")
            print("-"*40)
            print(f"トレンド: {trend_var/total_var*100:.1f}%")
            print(f"季節性: {seasonal_var/total_var*100:.1f}%")
            print(f"残差: {resid_var/total_var*100:.1f}%")
        
        return self.decomposition
    
    def plot_decomposition(self, figsize=(12, 10)):
        """分解結果の可視化（改良版）"""
        if not hasattr(self, 'decomposition'):
            print("先にseasonal_decomposition()を実行してください")
            return
        
        # カスタム分解プロット
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # モデルタイプをタイトルに含める
        model_label = "Multiplicative" if self.model == 'multiplicative' else "Additive"
        
        # 1. 元データ
        axes[0].plot(self.df.index, self.observed, 
                    linewidth=1.5, color='darkblue', label='Observed')
        axes[0].set_ylabel('Original')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f'Seasonal Decomposition ({model_label} Model)', 
                         fontsize=14, fontweight='bold')
        
        # 2. トレンド
        axes[1].plot(self.df.index, self.trend, 
                    linewidth=2, color='red', label='Trend')
        axes[1].set_ylabel('Trend')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 季節性
        axes[2].plot(self.df.index, self.seasonal, 
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
        axes[3].scatter(self.df.index, self.resid, 
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
        
        # 標準のプロット（statsmodelsの組み込み）
        print("\n標準分解プロット:")
        self.decomposition.plot()
        plt.show()
        
    def analyze_seasonal_pattern(self):
        """季節パターンの詳細分析"""
        if not hasattr(self, 'seasonal'):
            print("先にseasonal_decomposition()を実行してください")
            return
        
        # 月別の季節成分を集計
        seasonal_df = pd.DataFrame({
            'month': self.df['month'],
            'seasonal': self.seasonal
        })
        
        monthly_seasonal = seasonal_df.groupby('month')['seasonal'].agg(['mean', 'std'])
        
        # グラフ作成
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 月別季節成分の平均値
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        if self.model == 'multiplicative':
            bars = axes[0, 0].bar(months, monthly_seasonal['mean'], color='steelblue')
            axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Base (1.0)')
            axes[0, 0].set_ylabel('Seasonal Effect (Multiplier)')
            
            # バーの上に値を表示
            for bar, value in zip(bars, monthly_seasonal['mean']):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            bars = axes[0, 0].bar(months, monthly_seasonal['mean'], color='steelblue')
            axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Base (0)')
            axes[0, 0].set_ylabel('Seasonal Effect')
            
            # バーの上に値を表示
            for bar, value in zip(bars, monthly_seasonal['mean']):
                height = bar.get_height()
                y_pos = height if height > 0 else height - abs(height) * 0.1
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., y_pos,
                              f'{value:.0f}', ha='center', 
                              va='bottom' if height > 0 else 'top', fontsize=8)
        
        axes[0, 0].set_title('Average Seasonal Component by Month', fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. 季節性の年次変化
        pivot_seasonal = pd.DataFrame({
            'month': self.df['month'],
            'year': self.df['year'],
            'seasonal': self.seasonal
        }).pivot(index='month', columns='year', values='seasonal')
        
        for year in pivot_seasonal.columns:
            axes[0, 1].plot(pivot_seasonal.index, pivot_seasonal[year], 
                          marker='o', label=f'{year}', alpha=0.7)
        
        if self.model == 'multiplicative':
            axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5)
        else:
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
        axes[0, 1].set_title('Seasonal Pattern by Year', fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Seasonal Effect')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(range(1, 13))
        
        # 3. 季節性指数
        if self.model == 'multiplicative':
            seasonal_index = monthly_seasonal['mean'] * 100
        else:
            base_value = self.df[self.value_col].mean()
            seasonal_index = ((monthly_seasonal['mean'] + base_value) / base_value) * 100
        
        axes[1, 0].plot(months, seasonal_index, marker='o', 
                       linewidth=2, markersize=8, color='darkgreen')
        axes[1, 0].axhline(y=100, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Seasonal Index (Base=100)', fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Index')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([seasonal_index.min() - 5, seasonal_index.max() + 5])
        
        # インデックス値を表示
        for i, (month, value) in enumerate(zip(months, seasonal_index)):
            axes[1, 0].text(i, value + 1, f'{value:.1f}', 
                          ha='center', fontsize=8)
        
        # 4. 季節性の強度ヒートマップ
        heatmap_data = pivot_seasonal.T
        
        im = axes[1, 1].imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r')
        axes[1, 1].set_xticks(range(12))
        axes[1, 1].set_xticklabels(months)
        axes[1, 1].set_yticks(range(len(heatmap_data)))
        axes[1, 1].set_yticklabels(heatmap_data.index)
        axes[1, 1].set_title('Seasonal Component Heatmap', fontweight='bold')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Year')
        
        # カラーバー追加
        plt.colorbar(im, ax=axes[1, 1])
        
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
        
        return monthly_seasonal, seasonal_index
    
    def trend_analysis(self, figsize=(14, 6)):
        """トレンド成分の詳細分析"""
        if not hasattr(self, 'trend'):
            print("先にseasonal_decomposition()を実行してください")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. トレンドと元データの比較
        axes[0].plot(self.df.index, self.observed, 
                    linewidth=1, alpha=0.5, label='Original', color='gray')
        axes[0].plot(self.df.index, self.trend, 
                    linewidth=2, label='Trend', color='red')
        axes[0].set_title('Original vs Trend', fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Revenue')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. トレンドの変化率
        trend_pct_change = self.trend.pct_change() * 100
        axes[1].plot(self.df.index, trend_pct_change, 
                    linewidth=1, color='darkblue')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_title('Trend Growth Rate (%)', fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Growth Rate (%)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. トレンドの予測（単純な線形回帰）
        from sklearn.linear_model import LinearRegression
        
        X = np.arange(len(self.trend)).reshape(-1, 1)
        y = self.trend.fillna(method='ffill').fillna(method='bfill').values
        
        model = LinearRegression()
        model.fit(X, y)
        trend_pred = model.predict(X)
        
        axes[2].plot(self.df.index, self.trend, 
                    linewidth=2, label='Actual Trend', color='red')
        axes[2].plot(self.df.index, trend_pred, 
                    linewidth=1, linestyle='--', label='Linear Fit', color='blue')
        axes[2].set_title('Trend with Linear Fit', fontweight='bold')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Revenue')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # トレンド統計
        print("\n"+"="*60)
        print("トレンド分析結果")
        print("="*60)
        
        start_val = self.trend.dropna().iloc[0]
        end_val = self.trend.dropna().iloc[-1]
        total_growth = (end_val / start_val - 1) * 100
        
        print(f"開始時点の値: {start_val:,.0f}")
        print(f"終了時点の値: {end_val:,.0f}")
        print(f"期間全体の成長率: {total_growth:.1f}%")
        print(f"平均月次成長率: {trend_pct_change.dropna().mean():.2f}%")
        print(f"成長率の標準偏差: {trend_pct_change.dropna().std():.2f}%")
        print(f"線形回帰の傾き（月次）: {model.coef_[0]:,.0f}")
        
    def residual_diagnostics(self, lags=24):
        """残差の診断"""
        if not hasattr(self, 'resid'):
            print("先にseasonal_decomposition()を実行してください")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 残差データの準備
        resid_clean = self.resid.dropna()
        
        # 1. 残差の時系列プロット
        axes[0, 0].scatter(resid_clean.index, resid_clean.values, 
                         s=10, alpha=0.5, color='gray')
        if self.model == 'multiplicative':
            axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5)
        else:
            axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Residuals over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差のヒストグラム
        axes[0, 1].hist(resid_clean, bins=30, 
                       edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].set_title('Residual Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 正規性の検定結果を追加
        from scipy import stats
        _, p_value = stats.normaltest(resid_clean)
        axes[0, 1].text(0.05, 0.95, f'Normality test p-value: {p_value:.4f}',
                       transform=axes[0, 1].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. ACF
        plot_acf(resid_clean, lags=min(lags, len(resid_clean)//2), ax=axes[1, 0])
        axes[1, 0].set_title('Residual ACF', fontweight='bold')
        
        # 4. Q-Qプロット
        stats.probplot(resid_clean, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Ljung-Box検定
        lb_test = acorr_ljungbox(resid_clean, 
                                lags=min(12, len(resid_clean)//2), 
                                return_df=True)
        
        print("\n"+"="*60)
        print("残差診断結果")
        print("="*60)
        
        if self.model == 'multiplicative':
            print(f"残差の平均（理想値=1.0）: {resid_clean.mean():.4f}")
            print(f"残差の標準偏差: {resid_clean.std():.4f}")
        else:
            print(f"残差の平均（理想値=0.0）: {resid_clean.mean():.4f}")
            print(f"残差の標準偏差: {resid_clean.std():.2f}")
            
        print(f"残差の歪度: {resid_clean.skew():.4f}")
        print(f"残差の尖度: {resid_clean.kurtosis():.4f}")
        
        print("\nLjung-Box検定（残差の自己相関）:")
        print("-"*40)
        print(lb_test[['lb_stat', 'lb_pvalue']].tail(1))
        
        if lb_test['lb_pvalue'].iloc[-1] > 0.05:
            print("\n結果: 残差に有意な自己相関なし（モデル適合良好）")
        else:
            print("\n結果: 残差に有意な自己相関あり（モデル改善の余地あり）")
            
        if p_value > 0.05:
            print("正規性検定: 残差は正規分布に従う（p > 0.05）")
        else:
            print("正規性検定: 残差は正規分布に従わない（p < 0.05）")
    
    def compare_models(self, period=12):
        """加法モデルと乗法モデルの比較"""
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # 加法モデル
        decomp_add = seasonal_decompose(
            self.df[self.value_col], 
            model='additive', 
            period=period,
            extrapolate_trend='freq'
        )
        
        # 乗法モデル
        decomp_mult = seasonal_decompose(
            self.df[self.value_col], 
            model='multiplicative', 
            period=period,
            extrapolate_trend='freq'
        )
        
        # 加法モデルのプロット
        axes[0, 0].plot(decomp_add.trend, linewidth=2, color='red')
        axes[0, 0].set_title('Additive - Trend', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylabel('Value')
        
        axes[1, 0].plot(decomp_add.seasonal, linewidth=1, color='green')
        axes[1, 0].set_title('Additive - Seasonal', fontweight='bold')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylabel('Value')
        
        axes[2, 0].scatter(decomp_add.resid.index, decomp_add.resid.values, 
                         s=5, alpha=0.5, color='gray')
        axes[2, 0].set_title('Additive - Residual', fontweight='bold')
        axes[2, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_ylabel('Value')
        axes[2, 0].set_xlabel('Date')
        
        # 乗法モデルのプロット
        axes[0, 1].plot(decomp_mult.trend, linewidth=2, color='red')
        axes[0, 1].set_title('Multiplicative - Trend', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylabel('Value')
        
        axes[1, 1].plot(decomp_mult.seasonal, linewidth=1, color='green')
        axes[1, 1].set_title('Multiplicative - Seasonal', fontweight='bold')
        axes[1, 1].axhline(y=1, color='black', linestyle='--', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylabel('Multiplier')
        
        axes[2, 1].scatter(decomp_mult.resid.index, decomp_mult.resid.values, 
                         s=5, alpha=0.5, color='gray')
        axes[2, 1].set_title('Multiplicative - Residual', fontweight='bold')
        axes[2, 1].axhline(y=1, color='black', linestyle='--', alpha=0.3)
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_ylabel('Multiplier')
        axes[2, 1].set_xlabel('Date')
        
        plt.tight_layout()
        plt.show()
        
        # モデル比較の統計
        resid_add_var = decomp_add.resid.dropna().var()
        resid_mult_var = decomp_mult.resid.dropna().var()
        
        # AICの簡易計算（残差の分散ベース）
        n = len(decomp_add.resid.dropna())
        aic_add = n * np.log(resid_add_var) + 2 * 3  # 3 components
        aic_mult = n * np.log(resid_mult_var) + 2 * 3
        
        print("\n"+"="*60)
        print("加法モデル vs 乗法モデルの比較")
        print("="*60)
        print("\n残差の分散:")
        print(f"加法モデル: {resid_add_var:,.2f}")
        print(f"乗法モデル: {resid_mult_var:.6f}")
        
        print("\nAIC（赤池情報量基準）:")
        print(f"加法モデル: {aic_add:.2f}")
        print(f"乗法モデル: {aic_mult:.2f}")
        
        if aic_add < aic_mult:
            print("\n推奨: 加法モデル（AICが小さい）")
        else:
            print("\n推奨: 乗法モデル（AICが小さい）")
        
        print("\nモデル選択のガイドライン:")
        print("-"*40)
        print("加法モデルが適している場合:")
        print("- 季節変動が時間とともに一定")
        print("- トレンドと季節性が独立")
        print("\n乗法モデルが適している場合:")
        print("- 季節変動がデータレベルに比例")
        print("- 成長率が一定の時系列")
    
    def forecast_next_period(self, periods=12):
        """簡単な予測（トレンド延長 + 季節性）"""
        if not hasattr(self, 'decomposition'):
            print("先にseasonal_decomposition()を実行してください")
            return
        
        # トレンドの線形延長
        from sklearn.linear_model import LinearRegression
        
        # トレンドデータの準備
        trend_clean = self.trend.dropna()
        X = np.arange(len(trend_clean)).reshape(-1, 1)
        y = trend_clean.values
        
        # 線形モデルのフィット
        model = LinearRegression()
        model.fit(X, y)
        
        # 将来の予測
        future_X = np.arange(len(trend_clean), len(trend_clean) + periods).reshape(-1, 1)
        future_trend = model.predict(future_X)
        
        # 季節パターンの繰り返し
        seasonal_pattern = self.seasonal[:12].values
        future_seasonal = np.tile(seasonal_pattern, (periods // 12) + 1)[:periods]
        
        # 予測値の計算
        if self.model == 'multiplicative':
            forecast = future_trend * future_seasonal
        else:
            forecast = future_trend + future_seasonal
        
        # 予測日付の生成
        last_date = self.df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                    periods=periods, freq='M')
        
        # グラフ化
        plt.figure(figsize=(14, 6))
        
        # 実績データ
        plt.plot(self.df.index, self.observed, 
                linewidth=2, label='Actual', color='darkblue')
        
        # 予測データ
        plt.plot(future_dates, forecast, 
                linewidth=2, linestyle='--', label='Forecast', color='red')
        
        # 信頼区間（簡易版）
        resid_std = self.resid.dropna().std()
        if self.model == 'multiplicative':
            upper_bound = forecast * (1 + 2 * resid_std)
            lower_bound = forecast * (1 - 2 * resid_std)
        else:
            upper_bound = forecast + 2 * resid_std
            lower_bound = forecast - 2 * resid_std
        
        plt.fill_between(future_dates, lower_bound, upper_bound, 
                        alpha=0.3, color='red', label='95% CI')
        
        plt.title(f'Revenue Forecast - Next {periods} Months ({self.model.capitalize()} Model)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Revenue')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 予測結果の表示
        print("\n"+"="*60)
        print(f"予測結果（次{periods}ヶ月）")
        print("="*60)
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast,
            'Lower_95CI': lower_bound,
            'Upper_95CI': upper_bound
        })
        
        print(forecast_df.head())
        print(f"\n予測期間の平均: {forecast.mean():,.0f}")
        print(f"予測期間の最大値: {forecast.max():,.0f}")
        print(f"予測期間の最小値: {forecast.min():,.0f}")
        
        return forecast_df

# サンプルデータ生成関数
def generate_sample_data(model_type='multiplicative'):
    """サンプル収益データの生成"""
    np.random.seed(42)
    
    # 4年分の月次データ
    dates = pd.date_range(start='2021-01-01', end='2024-12-31', freq='M')
    
    if model_type == 'multiplicative':
        # 乗法モデル用データ
        trend = np.linspace(100000, 200000, len(dates))
        
        # 季節成分（乗数）
        seasonal_pattern = [
            0.90, 0.85, 0.95, 1.00, 1.05, 1.10,  # 1-6月
            1.15, 1.12, 1.08, 1.05, 1.20, 1.35   # 7-12月
        ]
        seasonal = np.tile(seasonal_pattern, len(dates)//12 + 1)[:len(dates)]
        
        # ノイズ（乗数）
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
    print("seasonal_decomposeによる時系列分析")
    print("加法モデル・乗法モデル完全対応版")
    print("="*60)
    print()
    
    # 1. 乗法モデルでの分析
    print("\n【Part 1: 乗法モデルでの分析】")
    print("="*60)
    
    # データの準備
    print("サンプルデータを生成中（乗法モデル用）...")
    df_mult = generate_sample_data(model_type='multiplicative')
    
    # 分析クラスのインスタンス化（乗法モデル）
    analyzer_mult = SeasonalAnalyzer(
        df_mult, 
        date_col='date', 
        value_col='revenue', 
        model='multiplicative'
    )
    
    # 基本的な可視化
    print("\n1. 基本的な時系列グラフ")
    analyzer_mult.basic_visualization()
    
    # 時系列分解
    print("\n2. 時系列分解（乗法モデル）")
    decomp_mult = analyzer_mult.seasonal_decomposition(period=12)
    
    # 分解結果の可視化
    print("\n3. 分解結果の可視化")
    analyzer_mult.plot_decomposition()
    
    # 季節パターン分析
    print("\n4. 季節パターンの詳細分析")
    monthly_seasonal_mult, seasonal_index_mult = analyzer_mult.analyze_seasonal_pattern()
    
    # トレンド分析
    print("\n5. トレンド分析")
    analyzer_mult.trend_analysis()
    
    # 残差診断
    print("\n6. 残差の診断")
    analyzer_mult.residual_diagnostics()
    
    # モデル比較
    print("\n7. 加法モデルと乗法モデルの比較")
    analyzer_mult.compare_models()
    
    # 予測
    print("\n8. 将来予測")
    forecast_mult = analyzer_mult.forecast_next_period(periods=12)
    
    print("\n"+"="*60)
    print("\n【Part 2: 加法モデルでの分析（比較用）】")
    print("="*60)
    
    # 同じデータで加法モデルも試す
    analyzer_add = SeasonalAnalyzer(
        df_mult,  # 同じデータを使用
        date_col='date', 
        value_col='revenue', 
        model='additive'
    )
    
    print("\n加法モデルでの時系列分解")
    decomp_add = analyzer_add.seasonal_decomposition(period=12)
    analyzer_add.plot_decomposition()
    
    print("\n"+"="*60)
    print("分析完了")
    print("="*60)
    print("\nまとめ:")
    print("- seasonal_decomposeは乗法モデルを直接サポート")
    print("- model='multiplicative'で簡単に切り替え可能")
    print("- 季節変動がトレンドに比例する場合は乗法モデルが有効")
    print("- AICなどの指標でモデル選択が可能")
```
