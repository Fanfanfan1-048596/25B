import numpy as np
from math import log, pi
from Q2_LR import read_spectrum, n2_4H, n2_6H, compute_X, extract_peaks
import warnings
warnings.filterwarnings('ignore')

# 先验与似然函数
log_prior = lambda d, s: -0.5*((d-9e-4)/2e-4)**2 - 3*log(s) - 1/s
log_likelihood = lambda d, b, s, X, M: -0.5*len(M)*log(2*pi*s**2) - 0.5*np.sum((M-d*X-b)**2)/s**2
log_posterior = lambda params, X, M: -np.inf if params[0]<=0 or params[2]<=0 else log_likelihood(*params, X, M) + log_prior(params[0], params[2])

def mcmc_sample(X, M, n_samples=10000, burn_in=2000):
    """MCMC采样"""
    params = np.array([9e-4, 0.0, 0.1])  # [d, b, sigma]
    cov = np.diag([1e-8, 1.0, 0.01])
    samples, n_accepted = [], 0
    current_log_post = log_posterior(params, X, M)
    
    for i in range(n_samples + burn_in):
        proposal = np.random.multivariate_normal(params, cov)
        proposal_log_post = log_posterior(proposal, X, M)
        if proposal_log_post - current_log_post > 0 or np.random.random() < np.exp(proposal_log_post - current_log_post):
            params, current_log_post = proposal, proposal_log_post
            n_accepted += 1
        if i >= burn_in: samples.append(params.copy())
    
    return np.array(samples), n_accepted / (n_samples + burn_in)

def bayesian_single(X, M):
    """单数据集贝叶斯分析"""
    samples, acceptance_rate = mcmc_sample(X, M)
    d_samples = samples[:, 0] * 1e4  # 转换为微米
    d_mean, d_std = np.mean(d_samples), np.std(d_samples)
    d_ci = np.percentile(d_samples, [2.5, 97.5])
    M_pred = np.mean(samples[:, 0:1] * X + samples[:, 1:2], axis=0)
    r2 = 1 - np.var(M - M_pred) / np.var(M)
    return {'d_mean': d_mean, 'd_std': d_std, 'd_ci': d_ci, 'r2': r2, 'acceptance_rate': acceptance_rate}

def joint_bayesian(X1, M1, X2, M2):
    """联合贝叶斯分析"""
    joint_log_post = lambda params: -np.inf if params[0]<=0 or params[3]<=0 or params[4]<=0 else (
        log_likelihood(params[0], params[1], params[3], X1, M1) + 
        log_likelihood(params[0], params[2], params[4], X2, M2) + 
        log_prior(params[0], params[3]) + log_prior(params[0], params[4]) - log_prior(params[0], params[3])  # 避免重复计算d的先验
    )
    
    params = np.array([9e-4, 0.0, 0.0, 0.1, 0.1])  # [d, b1, b2, sigma1, sigma2]
    cov = np.diag([1e-8, 1.0, 1.0, 0.01, 0.01])
    samples, n_accepted = [], 0
    current_log_post = joint_log_post(params)
    
    for i in range(18000):  # 15000 + 3000 burn-in
        proposal = np.random.multivariate_normal(params, cov)
        proposal_log_post = joint_log_post(proposal)
        if proposal_log_post - current_log_post > 0 or np.random.random() < np.exp(proposal_log_post - current_log_post):
            params, current_log_post = proposal, proposal_log_post
            n_accepted += 1
        if i >= 3000: samples.append(params.copy())
    
    samples = np.array(samples)
    d_samples = samples[:, 0] * 1e4
    return {
        'd_mean': np.mean(d_samples), 'd_std': np.std(d_samples), 
        'd_ci': np.percentile(d_samples, [2.5, 97.5]),
        'acceptance_rate': n_accepted / 18000
    }

def analyze_bayesian(file1_path, theta1, file2_path=None, theta2=None, prominence=0.002, distance=10):
    """贝叶斯推断分析"""
    models = {'4H': n2_4H, '6H': n2_6H}
    results = {}
    
    print("贝叶斯MCMC厚度估算")
    
    for name, n2_fn in models.items():
        print(f'\n{name}-SiC折射率模型:')
        
        nu1, R1 = read_spectrum(file1_path)
        peaks1 = extract_peaks(nu1, R1, prominence, distance)
        X1, _ = compute_X(peaks1, theta1, n2_fn)
        M1 = np.arange(1, len(X1)+1, dtype=float)
        
        res1 = bayesian_single(X1, M1)
        print(f"附件1计算结果: {res1['d_mean']:.3f}±{res1['d_std']:.3f}μm, 95%CI[{res1['d_ci'][0]:.3f},{res1['d_ci'][1]:.3f}], R^2={res1['r2']:.3f}, {len(X1)}个峰")
        
        if file2_path and theta2:
            nu2, R2 = read_spectrum(file2_path)
            peaks2 = extract_peaks(nu2, R2, prominence, distance)
            X2, _ = compute_X(peaks2, theta2, n2_fn)
            M2 = np.arange(1, len(X2)+1, dtype=float)
            
            res2 = bayesian_single(X2, M2)
            print(f"附件2计算结果: {res2['d_mean']:.3f}±{res2['d_std']:.3f}μm, 95%CI[{res2['d_ci'][0]:.3f},{res2['d_ci'][1]:.3f}], R^2={res2['r2']:.3f}, {len(X2)}个峰")

            joint_res = joint_bayesian(X1, M1, X2, M2)
            print(f"联合推断结果: {joint_res['d_mean']:.3f}±{joint_res['d_std']:.3f}μm, CI[{joint_res['d_ci'][0]:.3f},{joint_res['d_ci'][1]:.3f}]")
            
            # 一致性检验
            overlap = len(set(range(int(res1['d_ci'][0]*100), int(res1['d_ci'][1]*100))) & 
                         set(range(int(res2['d_ci'][0]*100), int(res2['d_ci'][1]*100)))) / 100
            consistency = "好" if overlap > 0.5 else "一般" if overlap > 0.2 else "差"
            print(f"两角度数据一致性: {consistency}")
            
            results[name] = {'single_1': res1, 'single_2': res2, 'joint': joint_res, 'consistency': consistency}
        else:
            results[name] = {'single_1': res1}
    
    return results

if __name__ == '__main__':
    np.random.seed(42)
    results = analyze_bayesian("附件/附件1.xlsx", 10, "附件/附件2.xlsx", 15)
    
    print("\n最终结果:")
    for model in ['4H', '6H']:
        if model in results and 'joint' in results[model]:
            joint = results[model]['joint']
            print(f"{model}: {joint['d_mean']:.3f}±{joint['d_std']:.3f}μm (置信区间{joint['d_ci'][0]:.3f}-{joint['d_ci'][1]:.3f})")
