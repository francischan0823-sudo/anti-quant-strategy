import tushare as ts
import pandas as pd
import numpy as np
import datetime
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import time

import os

# 配置信息
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', 'f74d16b5935f9b88c0a832f38c0fe183a620ea92c2e6056076dcca07')
EMAIL_SENDER = '3674477@qq.com'
EMAIL_AUTH_CODE = os.getenv('EMAIL_AUTH_CODE', 'zmjamtrdnsrsbjif')
EMAIL_RECEIVER = '3674477@qq.com'

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

def get_trading_days():
    """获取最近的交易日：周五和周一"""
    # 生产环境逻辑：获取最近的一个周一和它之前的交易日
    today = datetime.datetime.now()
    # 如果今天是周一，则 monday 是今天，friday 是上一个交易日
    # 如果今天不是周一，则寻找最近的一个周一
    days_to_monday = (today.weekday() - 0) % 7
    monday_dt = today - datetime.timedelta(days=days_to_monday)
    monday = monday_dt.strftime('%Y%m%d')
    
    # 获取交易日历
    df_cal = pro.trade_cal(exchange='', is_open='1', start_date=(monday_dt - datetime.timedelta(days=10)).strftime('%Y%m%d'), end_date=monday)
    trading_days = df_cal['cal_date'].tolist()
    if monday in trading_days:
        idx = trading_days.index(monday)
        friday = trading_days[idx-1]
    else:
        # 如果周一不是交易日，则取上一个交易日作为 monday (这可能不符合逻辑，但作为兜底)
        monday = trading_days[-1]
        friday = trading_days[-2]
        
    return friday, monday

def get_bollinger_bands(df, n=20, k=2):
    """计算布林带"""
    df['MA'] = df['close'].rolling(window=n).mean()
    df['STD'] = df['close'].rolling(window=n).std()
    df['upper'] = df['MA'] + (k * df['STD'])
    df['lower'] = df['MA'] - (k * df['STD'])
    return df

from openai import OpenAI

def check_sentiment(stock_name, industry, date_range):
    """
    利用 LLM 分析该股票或行业在指定日期范围内的舆情热度。
    """
    client = OpenAI() # 使用预配置的客户端
    
    query = f"{date_range} {stock_name} {industry} 财经新闻 利好 题材发酵"
    # 注意：在 GitHub Actions 中，我们无法直接调用外部搜索工具，
    # 但我们可以通过 LLM 的知识库或模拟搜索结果。
    # 为了演示真实逻辑，这里构建一个 Prompt 让 LLM 评估该题材。
    
    prompt = f"""
    请分析在 {date_range} 期间，关于“{stock_name}”所属的“{industry}”板块是否有显著的题材发酵或利好新闻。
    
    评价标准：
    1. 是否有政策利好？
    2. 是否有行业重大突破或新闻？
    3. 社交媒体或财经媒体讨论热度是否显著上升？
    
    请给出 0-100 的评分，并简要说明理由。
    格式：评分: [数字] | 理由: [简短描述]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        score_part = content.split('|')[0].replace('评分:', '').strip()
        score = int(score_part)
        reason = content.split('|')[1].replace('理由:', '').strip()
        return score, reason
    except Exception as e:
        print(f"舆情分析出错: {e}")
        return 50, "分析失败，默认中性"

def select_stocks():
    friday, monday = get_trading_days()
    print(f"分析日期: 周五({friday}), 周一({monday})")
    
    # 1. 获取周五异动板块（涨幅靠前的概念或行业）
    # 这里简化为获取所有股票周五的表现
    df_friday = pro.daily(trade_date=friday)
    if df_friday.empty:
        return "未获取到周五数据"
    
    # 筛选周五涨幅大于5%的股票作为“异动”代表
    active_stocks = df_friday[df_friday['pct_chg'] > 5]['ts_code'].tolist()
    
    results = []
    
    # 限制测试数量
    print(f"待处理股票数量: {len(active_stocks[:50])}")
    for i, ts_code in enumerate(active_stocks[:50]):
        if i % 10 == 0: print(f"正在处理第 {i} 只股票...")
        try:
            # 获取个股历史数据计算布林带（需要至少20天数据）
            end_date = monday
            start_date = (datetime.datetime.strptime(monday, '%Y%m%d') - datetime.timedelta(days=60)).strftime('%Y%m%d')
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df.empty or len(df) < 20: continue
            df = df.sort_values('trade_date')
            df = get_bollinger_bands(df)
            
            # 确保包含周五和周一的数据
            df_friday = df[df['trade_date'] == friday]
            df_monday = df[df['trade_date'] == monday]
            
            if df_friday.empty or df_monday.empty:
                continue
                
            row_friday = df_friday.iloc[0]
            row_monday = df_monday.iloc[0]
            
            # 条件筛选：
            # 1. 周五：股价站上布林带上轨
            cond1 = row_friday['close'] > row_friday['upper']
            
            # 2. 周一：高开低走 (open > close 且 open > row_friday['close'])
            cond2 = row_monday['open'] > row_monday['close'] and row_monday['open'] > row_friday['close']
            
            # 3. 周一：缩量回调 (volume_monday < volume_friday)
            cond3 = row_monday['vol'] < row_friday['vol']
            
            # 4. 周一：最低价不跌破布林带上轨
            cond4 = row_monday['low'] >= row_monday['upper']
            
            # 5. 动量向上 (简单判断：周五涨幅 > 0)
            cond5 = row_friday['pct_chg'] > 0
            
            if cond1 and cond2 and cond3 and cond4 and cond5:
                stock_info = pro.stock_basic(ts_code=ts_code, fields='name,industry')
                name = stock_info.iloc[0]['name']
                industry = stock_info.iloc[0]['industry']
                
                # 增加舆情因子校验
                date_range = f"{friday}至{monday}"
                sentiment_score, reason = check_sentiment(name, industry, date_range)
                
                if sentiment_score >= 70: # 舆情评分阈值
                    print(f"找到符合条件的股票(含舆情发酵): {name} ({ts_code}), 评分: {sentiment_score}")
                    results.append({
                        '代码': ts_code,
                        '名称': name,
                        '行业': industry,
                        '周五收盘': row_friday['close'],
                        '周一开盘': row_monday['open'],
                        '周一收盘': row_monday['close'],
                        '周一最低': row_monday['low'],
                        '上轨线': round(row_monday['upper'], 2),
                        '舆情评分': sentiment_score,
                        '发酵理由': reason
                    })
            else:
                # 打印不符合的原因（可选，用于调试）
                pass
        except Exception as e:
            print(f"处理 {ts_code} 出错: {e}")
            continue
            
    return results

def send_email(content):
    if not content:
        content = "今日未筛选出符合条件的个股。"
    else:
        # 格式化表格
        df_res = pd.DataFrame(content)
        content = "符合反量化策略的个股列表：\n\n" + df_res.to_markdown()
        
    msg = MIMEText(content, 'plain', 'utf-8')
    msg['From'] = Header("量化监控助手", 'utf-8')
    msg['To'] = Header("投资者", 'utf-8')
    msg['Subject'] = Header("反量化策略选股结果", 'utf-8')
    
    try:
        server = smtplib.SMTP_SSL("smtp.qq.com", 465)
        server.login(EMAIL_SENDER, EMAIL_AUTH_CODE)
        server.sendmail(EMAIL_SENDER, [EMAIL_RECEIVER], msg.as_string())
        server.quit()
        print("邮件发送成功")
    except Exception as e:
        print(f"邮件发送失败: {e}")

if __name__ == "__main__":
    # 模拟运行
    stocks = select_stocks()
    if isinstance(stocks, list):
        print(f"筛选出 {len(stocks)} 只个股")
        send_email(stocks)
    else:
        print(stocks)
