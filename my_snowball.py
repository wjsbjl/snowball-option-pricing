import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import trange
from datetime import datetime
import os
if not os.path.exists('./result/'):   #os：operating system，包含操作系统功能，可以进行文件操作
    os.mkdir('./result/') #如果存在那就是这个result_path，如果不存在那就新建一个
if os.name == 'posix': # 如果系统是mac或者linux
    plt.rcParams['font.sans-serif'] = ['Songti SC'] #中文字体为宋体
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 在windows系统下显示微软雅黑
plt.rcParams['axes.unicode_minus'] = False # 负号用 ASCII 编码的-显示，而不是unicode的 U+2212
# pd.set_option('display.max_rows', 20)
# pd.set_option('display.max_columns', 10)

class my_plot():  # 后面再封装一些其他函数
    def __init__(self, plot_df, plot_name):
        self.plot_df = plot_df
        self.plot_name = plot_name

    def line_plot(self):  # name包括title，xlabel，ylabel，save_name
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        fontsize = 12
        x = self.plot_df.index
        y_labels = self.plot_df.columns
        for i in trange(len(y_labels)):
            clmn = y_labels[i]
            axes.plot(x, self.plot_df.loc[:, clmn].values, label=clmn)
        axes.set_title(f'{self.plot_name[0]}', fontsize=fontsize)
        axes.set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
        axes.set_ylabel(f'{self.plot_name[2]}', fontsize=fontsize)
        axes.grid()
        # plt.legend()
        # plt.legend(fontsize = fontsize * 0.7, loc = 'best',ncol = 10,bbox_to_anchor=(1.03, -0.25))
        # axes.set_xticks( range(0,21,2) )
        # axes.set_xticklabels( [i for i in axes.get_xticks()], rotation=0 )
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)
        plt.savefig(f"./result/{self.plot_name[3]}_line_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)
        # plt.show()
    def hist_plot(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        fontsize = 12
        ax.hist(self.plot_df,bins = 20)
        ax.set_title(f'{self.plot_name[0]}', fontsize=fontsize)
        ax.set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
        ax.set_ylabel(f'{self.plot_name[2]}', fontsize=fontsize)
        plt.savefig(f"./result/{self.plot_name[3]}_hist_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)
    
    def quantiles_plot(self):
        qtl_df = self.plot_df
        qtl_df.dropna(inplace=True)
        qtl_df = qtl_df.rank() / len(qtl_df)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        fontsize = 12

        ax.scatter(qtl_df['x'], qtl_df['y'], s=0.00001)
        ax.set_title(f'{self.plot_name[0]}', fontsize=fontsize)
        ax.set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
        ax.set_ylabel(f'{self.plot_name[2]}', fontsize=fontsize)
        ax.grid()
        # plt.legend(fontsize = fontsize * 0.7, loc = 'best',ncol = 10,bbox_to_anchor=(1.03, -0.25))
        # ax.set_xticks( range(0,21,2) )
        # ax.set_xticklabels( [i for i in ax.get_xticks()], rotation=0 )
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)
        plt.savefig(f"./result/{self.plot_name[3]}_quantiles_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)
        # plt.show()
    
    def bar_plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        hist_x = [str(x) for x in self.index]
        hist_y = self.values
        ax.bar(hist_x, hist_y)
        ax.set_xlabel('逐笔交易对应时间',fontsize = 30)
        ax.set_ylabel('频数',fontsize = 30)
        plt.savefig(f"./result/{self.plot_name[3]}_bar_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)
        
class my_snowball(object):
    def __init__(self,S0, position, margin, sigma, r_riskfree, q, knock_out_coupon, hold_coupon, T_years, num_simulations, knock_in_out_df,start_date):
        self.S0 = S0
        self.position = position
        self.margin=margin
        self.sigma=sigma
        self.r_riskfree=r_riskfree
        self.q=q
        self.knock_out_coupon=knock_out_coupon
        self.hold_coupon=hold_coupon
        self.T_years=T_years
        self.knock_in_out_df=knock_in_out_df
        self.start_date = pd.to_datetime(start_date)
        self.num_simulations = int(num_simulations)
        
        end_date = self.start_date + pd.DateOffset(years=T_years)
        self.date_index = pd.date_range(start=start_date, end=end_date, freq='D')
        
        self.stock_price_array = self.simulation(steps = 365)
        self.stock_price_df = pd.DataFrame(self.stock_price_array, index = self.date_index)

        self.snowball_status = self.price2status()

        self.payoff_lst, self.time_out_lst, self.knock_status_lst = self.snowball_payoff()

        self.snowball_sttc_df = self.snowball_sttc()

    def simulation(self,  steps = 365):
        delta_t = 1/steps
        Spath = np.zeros((self.T_years * steps + 1, self.num_simulations))
        Spath[0,:] = self.S0

        for t in trange(1, self.T_years * steps + 1):
            z = np.random.standard_normal(self.num_simulations)
            middle1 = Spath[t-1, 0:self.num_simulations] * np.exp((self.r_riskfree - self.q - 0.5 * self.sigma ** 2) * delta_t + self.sigma * np.sqrt(delta_t) * z)
            uplimit = Spath[t-1,:] * 1.1 # 涨幅限制
            lowlimit = Spath[t-1,:] * 0.9 # 跌幅限制
            temp = np.where(uplimit < middle1, uplimit, middle1)
            temp = np.where(lowlimit > middle1, lowlimit, temp)
            Spath[t, 0:self.num_simulations] = temp

        return Spath

    def stock_price_line_plot(self):
        _ = my_plot(self.stock_price_df,['Stock Price Simulation','Date','Stock Price','stock_price'])
        _.line_plot()
        
    def stock_price_hist_plot(self):
        _ = my_plot(self.stock_price_df.iloc[-1,:],['Stock Price at the Expiry Date','Price','Frequency','stock_price'])
        _.hist_plot()
        
    def price2status(self):
        snow_price_df = self.stock_price_df.loc[self.knock_in_out_df.index,:]
        snow_df = snow_price_df.copy()
        snow_price_df['knock_out'] = self.knock_in_out_df['knock_out_lst']
        snow_price_df['knock_in'] = self.knock_in_out_df['knock_in_lst']
        snow_df[(snow_df.T > snow_price_df['knock_out']).T] = 1
        snow_df[(snow_df.T < snow_price_df['knock_in']).T] = -1
        snow_df[abs(snow_df) != 1] = 0
        return snow_df

    def snowball_payoff(self):
        payoff_lst = []
        time_out_lst = []
        knock_status_lst = []
        clmn_lst = self.snowball_status.columns
        # TODO 这里应该可以提高效率
        for i in trange(len(clmn_lst)):
            clmn = clmn_lst[i]
            if self.snowball_status.loc[:,clmn].max() == 1: # 敲出吃利息
                time_delta = self.snowball_status.loc[:,clmn].idxmax() - self.start_date# 从这个时候贴现回来
                time_out = time_delta.days/365
                payoff = (self.knock_out_coupon * time_out) * np.exp(-self.r_riskfree * time_out) * self.margin
                knock_status = '敲出'
            elif self.snowball_status.loc[:,clmn].min() == -1: # 敲入赔期权
                time_out = self.T_years
                payoff = min(self.stock_price_df.loc[:,clmn][-1]-self.S0,0) * np.exp(-self.r_riskfree * time_out) * self.margin
                knock_status = '敲入'
            else:
                time_out = self.T_years
                payoff = (self.hold_coupon * time_out) * np.exp(-self.r_riskfree * time_out) * self.margin
                knock_status = '未敲入敲出'
            payoff_lst.append(payoff)
            time_out_lst.append(time_out)
            knock_status_lst.append(knock_status)
        return payoff_lst,time_out_lst, knock_status_lst
    
    def snowball_sttc(self):
        snowball_df = pd.DataFrame()
        snowball_df['payoff'] = self.payoff_lst
        snowball_df['time (Year)'] = self.time_out_lst
        snowball_df['time (Month)'] = snowball_df['time (Year)'] * 12
        snowball_df['knock_status'] = self.knock_status_lst
        self.snowball_df = snowball_df
        snowball_descriptive_df = snowball_df.groupby('knock_status').mean()
        snowball_distribution_df = (snowball_df['knock_status'].value_counts()/len(snowball_df)).rename('percent')
        return pd.concat([snowball_descriptive_df,snowball_distribution_df],axis=1).round(4)


    def snowball_payoff_hist_plot(self):
        _ = my_plot(self.snowball_df['payoff'],['Payoff of the Snowball Option','Price','Frequency','payoff'])
        _.hist_plot()
        
if __name__ == '__main__':
    S0 = 1
    position = 1e8
    margin = 1
    sigma = 0.16
    r_riskfree = 0.03
    q = 0.095
    knock_out_coupon = 0.16
    hold_coupon = 0.16
    T_years = 2
    num_simulations = 3e5

    start_date = pd.to_datetime('2021-01-05') + pd.DateOffset(months=4)
    end_date = start_date + pd.DateOffset(years=T_years)
    date_lst = pd.date_range(start=start_date, end=end_date, freq='30D')

    knock_in_barrier = 0.75
    knock_out_barrier = 1.03

    knock_in_out_df = pd.DataFrame(index = date_lst, data = {'knock_in_lst': knock_in_barrier, 'knock_out_lst': knock_out_barrier})
    
    snowball_sim = my_snowball(S0, position, margin, sigma, r_riskfree, q, knock_out_coupon, hold_coupon, T_years, knock_in_out_df,start_date)
    
    snowball_sim.stock_price_line_plot()
    snowball_sim.stock_price_hist_plot()
    snowball_sim.snowball_payoff_hist_plot()
    snowball_sim