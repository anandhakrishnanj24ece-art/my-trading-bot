import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

def run_audit(name):
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
    data = yf.download(tickers, period='1y', interval='1d', progress=False)
    close = data['Close'].ffill().dropna()
    vol = data['Volume'].ffill().dropna()
    
    rets = close.pct_change().dropna()
    common = rets.index.intersection(close.index).intersection(vol.index)
    rets, close, vol = rets.loc[common], close.loc[common], vol.loc[common]
    
    alpha = LiquidityAlpha()
    exec_engine = StealthExecution()
    
    p_val = [1.0]
    curr_w = np.zeros(len(tickers))
    daily_rets = []

    for i in range(len(rets)):
        signals, confs = [], []
        for t in tickers:
            df = pd.DataFrame({'Close': close[t], 'Volume': vol[t]})
            res = alpha.generate_signal(df)
            if not res.empty and i < len(res):
                signals.append(res.iloc[i]['signal'])
                confs.append(res.iloc[i]['confidence'])
            else:
                signals.append(0.0); confs.append(0.0)
        
        s_arr, c_arr = np.array(signals), np.array(confs)
        
        if name == "Weighted":
            if np.abs(s_arr).sum() > 0:
                w = np.sign(s_arr) * (np.abs(s_arr) / np.abs(s_arr).sum())
            else: w = np.zeros(len(tickers))
            
            final_w_pre = []
            for j in range(len(w)):
                cap = 0.05 if c_arr[j] < 1.0 else (0.20 if c_arr[j] < 2.0 else 0.50)
                final_w_pre.append(np.clip(w[j], -cap, cap))
            final_w_pre = np.array(final_w_pre)
            if np.sum(np.abs(final_w_pre)) > 1.0: final_w_pre /= np.sum(np.abs(final_w_pre))
            
            risk = VolatilitySwitch(base_leverage=4.0)
            lev = risk.get_leverage(rets.iloc[:, 0].iloc[:i+1])
            final_w = final_w_pre * lev
        else: # v3
            # Simplified v3 logic
            snr = np.mean(np.abs(s_arr)) / (np.std(s_arr) + 1e-6)
            l_adj = 0.005 / (1.0 + snr)
            if np.abs(s_arr).sum() == 0: w = np.zeros(len(tickers))
            else:
                w = np.sign(s_arr) * (np.abs(s_arr) / np.abs(s_arr).sum())
                w = np.where(np.abs(w) > l_adj, w, 0)
                if np.abs(w).sum() > 0: w = w / np.abs(w).sum()
                else: w = np.zeros(len(tickers))
            
            final_w_pre = []
            for weight, conf in zip(w, c_arr):
                cap = 0.05 + (np.clip(conf, 0, 3) / 3.0) * (0.50 - 0.05)
                final_w_pre.append(np.clip(weight, -cap, cap))
            final_w_pre = np.array(final_w_pre)
            if np.sum(np.abs(final_w_pre)) > 1.0: final_w_pre /= np.sum(np.abs(final_w_pre))
            
            # v3 Risk
            recent = rets.tail(20) if i > 20 else rets.iloc[:i+1]
            try:
                cov = LedoitWolf().fit(recent).covariance_ * 252
            except: cov = recent.cov() * 252
            p_vol = np.sqrt(max(0, np.dot(final_w_pre.T, np.dot(cov, final_w_pre))))
            lev = min(0.15 / p_vol, 4.0) if p_vol > 0 else 4.0
            final_w = final_w_pre * lev

        day_ret = np.sum(final_w * rets.iloc[i].values)
        cost = exec_engine.calculate_cost(np.sum(np.abs(final_w - curr_w)))
        net = day_ret - cost
        daily_rets.append(net)
        p_val.append(p_val[-1] * (1 + net))
        curr_w = final_w

    s_rets = pd.Series(daily_rets)
    ann_ret = (np.prod(1 + s_rets) - 1)
    ann_vol = s_rets.std() * np.sqrt(252)
    sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0
    mdd = (1 - pd.Series(p_val).cummax() / pd.Series(p_val)).max()
    
    return {"Strategy": name, "Ret": f"{ann_ret*100:.2f}%", "Vol": f"{ann_vol*100:.2f}%", "Sharpe": f"{sharpe:.4f}", "MDD": f"{mdd*100:.2f}%"}

if __name__ == "__main__":
    print(pd.DataFrame([run_audit("Weighted"), run_audit("v3")]).to_string(index=False))
