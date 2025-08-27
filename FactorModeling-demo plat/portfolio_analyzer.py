import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# This analyzer is for after-cost clean portfolio 
class PortfolioAnalyzer:
    def __init__(self, df: pd.DataFrame, trading_days_per_year: int = 250):
        # Clean and set up data
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.sort_values('date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.trading_days = trading_days_per_year
        self.df['return'] = np.exp(self.df['log_return']) - 1  # Convert log return to simple return
        self.df['cumulative_return'] = (1 + self.df['return']).cumprod() - 1  # Start from 0%

        # --- Compute rolling Sharpe ratio as a new column ---
        window = 120  # rolling window in days
        excess = self.df['return']
        rolling_mean = excess.rolling(window).mean()
        rolling_std = excess.rolling(window).std()
        self.df['sharpe_ratio'] = rolling_mean / rolling_std * np.sqrt(self.trading_days)  # Annualized

    def average_return(self):
        return self.df['return'].mean()

    def daily_volatility(self):
        return self.df['return'].std()

    def yearly_volatility(self):
        return self.daily_volatility() * np.sqrt(self.trading_days)

    def annualized_return(self):
        total_days = (self.df['date'].iloc[-1] - self.df['date'].iloc[0]).days
        total_years = total_days / 365.25
        final_value = self.df['cumulative_return'].iloc[-1] + 1
        return final_value**(1 / total_years) - 1

    def sharpe_ratio(self, risk_free_rate=0):
        excess_return = self.df['return'] - risk_free_rate / self.trading_days
        return (excess_return.mean() / excess_return.std()) * np.sqrt(self.trading_days)

    def sortino_ratio(self, risk_free_rate=0):
        excess_return = self.df['return'] - risk_free_rate / self.trading_days
        downside_std = excess_return[excess_return < 0].std()
        return (excess_return.mean() / downside_std) * np.sqrt(self.trading_days)

    def max_drawdown(self):
        cum = self.df['cumulative_return'] + 1
        peak = cum.cummax()
        drawdown = cum / peak - 1
        return drawdown.min()

    def max_drawdown_curve(self):
        cum = self.df['cumulative_return'] + 1
        peak = cum.cummax()
        return cum / peak - 1

    def max_daily_return(self):
        return self.df['return'].max()

    def min_daily_return(self):
        return self.df['return'].min()

    def monthly_return(self):
        monthly = self.df.set_index('date')['return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        return monthly

    def yearly_return(self):
        yearly = self.df.set_index('date')['return'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
        return yearly

    def summary(self):
        return {
        'Average Daily Return': f"{round(self.average_return() * 100, 2)}%",
        'Annualized Return': f"{round(self.annualized_return() * 100, 2)}%",
        'Daily Volatility': f"{round(self.daily_volatility() * 100, 2)}%",
        'Yearly Volatility': f"{round(self.yearly_volatility() * 100, 2)}%",
        'Sharpe Ratio': round(self.sharpe_ratio(), 2),
        'Sortino Ratio': round(self.sortino_ratio(), 2),
        'Max Drawdown': f"{round(self.max_drawdown() * 100, 2)}%",
        'Max Daily Return': f"{round(self.max_daily_return() * 100, 2)}%",
        'Min Daily Return': f"{round(self.min_daily_return() * 100, 2)}%"
    }

    def plot_full_performance(self, counts_df: pd.DataFrame = None):
        # === Compute cumulative return columns ===
        self.df['cumulative_return'] = (self.df['log_return'].fillna(0)).cumsum().apply(np.exp) - 1

        if 'long_return' in self.df.columns:
            self.df['cumulative_long_return'] = self.df['long_return'].fillna(0).cumsum().apply(np.exp) - 1
        if 'short_return' in self.df.columns:
            self.df['cumulative_short_return'] = self.df['short_return'].fillna(0).cumsum().apply(np.exp) - 1

        monthly_return_df = (
            self.df.set_index('date')['return']
            .resample('ME')
            .apply(lambda x: (1 + x).prod() - 1)
            .dropna()
        )

        has_turnover = 'turnover' in self.df.columns
        has_counts = counts_df is not None and not counts_df.empty
        has_leg_returns = 'cumulative_long_return' in self.df.columns or 'cumulative_short_return' in self.df.columns
        has_leg_turnover = 'long_turnover' in self.df.columns or 'short_turnover' in self.df.columns

        n_rows = 4  # Base rows: main, hist/std, Sharpe, summary
        if has_turnover:
            n_rows += 1
        if has_counts:
            n_rows += 1

        fig = plt.figure(figsize=(14, 4 * n_rows))

        height_ratios = [2]
        if has_turnover:
            height_ratios.append(0.8)
        if has_counts:
            height_ratios.append(0.8)
        height_ratios.extend([1.2, 0.8, 0.5])  # 1.2 Hist/Std, 0.8 Sharpe, 0.5 Summary

        gs = GridSpec(n_rows, 2, height_ratios=height_ratios, figure=fig)

        ax_main = fig.add_subplot(gs[0, :])
        ax_ret = ax_main.twinx()

        ax_main.plot(self.df['date'], self.df['cumulative_return'], label='Total', color='black')
        ax_main.plot(self.df['date'], self.max_drawdown_curve(), label='Max Drawdown Curve', color='red', linestyle='--')

        if 'cumulative_long_return' in self.df.columns:
            ax_main.plot(self.df['date'], self.df['cumulative_long_return'], label='Long Leg', color='green', linestyle=':')
        if 'cumulative_short_return' in self.df.columns:
            ax_main.plot(self.df['date'], self.df['cumulative_short_return'], label='Short Leg', color='orange', linestyle='-.')

        ax_main.set_ylabel('Cumulative Return')
        ax_main.set_title('Cumulative Return (Total / Long / Short) with Monthly Bars')
        ax_main.legend(loc='upper left')
        ax_main.grid(True)

        bar_colors = ['green' if r >= 0 else 'red' for r in monthly_return_df]
        ax_ret.bar(monthly_return_df.index, monthly_return_df.values, width=20, color=bar_colors, alpha=0.4)
        ax_ret.set_ylabel('Monthly Return', color='gray')
        ax_ret.tick_params(axis='y', labelcolor='gray')

        current_row = 1

        if has_turnover:
            ax_turnover = fig.add_subplot(gs[current_row, :])
            ax_turnover.plot(self.df['date'], self.df['turnover'], color='purple', linewidth=1.2, label='Total Turnover')

            if has_leg_turnover:
                if 'long_turnover' in self.df.columns:
                    ax_turnover.plot(self.df['date'], self.df['long_turnover'], color='green', linestyle='--', label='Long Turnover')
                if 'short_turnover' in self.df.columns:
                    ax_turnover.plot(self.df['date'], self.df['short_turnover'], color='red', linestyle='--', label='Short Turnover')

            avg_turnover = self.df['turnover'].mean()
            ax_turnover.axhline(avg_turnover, color='gray', linestyle=':', linewidth=1.2, label=f'Avg: {avg_turnover:.2%}')

            ax_turnover.set_ylabel('Turnover')
            ax_turnover.set_title('Portfolio Turnover (Total / Long / Short)')
            ax_turnover.legend(loc='upper right')
            ax_turnover.grid(True)
            current_row += 1

        if has_counts:
            ax_counts = fig.add_subplot(gs[current_row, :])
            ax_counts.plot(counts_df.index, counts_df['long_count'], label='Long Count', color='green')
            ax_counts.plot(counts_df.index, counts_df['short_count'], label='Short Count', color='red')
            ax_counts.set_title("Number of Symbols in Long and Short Legs Over Time")
            ax_counts.set_xlabel("Date")
            ax_counts.set_ylabel("Count")
            ax_counts.legend()
            ax_counts.grid(True)
            current_row += 1

        ax_hist = fig.add_subplot(gs[current_row, 0])
        ax_hist.hist(self.df['return'].dropna(), bins=50, color='skyblue', edgecolor='k')
        ax_hist.set_title('Histogram of Daily Returns')
        ax_hist.set_xlabel('Daily Return')
        ax_hist.set_ylabel('Frequency')
        ax_hist.grid(True)

        ax_std = fig.add_subplot(gs[current_row, 1])
        daily_std = self.df['return'].dropna()
        standardized = (daily_std - daily_std.mean()) / daily_std.std()
        ax_std.hist(standardized, bins=50, color='orange', edgecolor='k', alpha=0.7)
        ax_std.set_title('Standardized Daily Return')
        ax_std.set_xlabel('Z-score')
        ax_std.set_ylabel('Frequency')
        ax_std.grid(True)

        current_row += 1

        # --- 120-Day Rolling Sharpe Ratio ---
        ax_sharpe = fig.add_subplot(gs[current_row, :])
        ax_sharpe.plot(self.df['date'], self.df['sharpe_ratio'], label='120d Sharpe', color='darkred')
        ax_sharpe.grid(True, axis='x', linestyle=':')
        ax_sharpe.set_title('120-Day Rolling Sharpe Ratio')
        ax_sharpe.legend()
        for y in ax_sharpe.get_yticks():
            if abs(y - round(y)) < 1e-6:  # highlight integer levels
                ax_sharpe.axhline(y, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

        current_row += 1

        ax_text = fig.add_subplot(gs[current_row, :])
        stats = self.summary()
        summary_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
        ax_text.axis('off')
        ax_text.text(0, 0.5, summary_text, fontsize=10, verticalalignment='center', fontfamily='monospace')

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Commented out for Replit compatibility - hardcoded path removed
    # benchmark_data = pd.read_csv("path/to/your/data.csv")
    # benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
    # benchmark_data = benchmark_data[benchmark_data['date'] >= '2020-01-01']
    # benchmark_data = benchmark_data[benchmark_data['date'] <= '2025-06-16']
    # 
    # benchmark_data = benchmark_data[['date', 'return']].reset_index(drop=True)
    # 
    # benchmark_data['return'] = benchmark_data['return'].astype(float)
    # benchmark_data['log_return'] = np.log(1 + benchmark_data['return'])
    # 
    # analyzer = PortfolioAnalyzer(benchmark_data)
    # analyzer.plot_full_performance()
    # 
    # summary_df = pd.DataFrame.from_dict(analyzer.summary(), orient='index', columns=['Value'])
    # summary_df.index.name = 'Metric'
    # summary_df.reset_index(inplace=True)
    # 
    # print(summary_df)
    pass