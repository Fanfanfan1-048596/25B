import numpy as np
from scipy.optimize import least_squares
from math import log
from Q2_LR import read_spectrum, n2_4H, n2_6H, compute_X, extract_peaks

# 非线性模型和拟合
nonlinear_model = lambda params, X: params[0]*X + params[1] + params[2]*X**2
residuals = lambda params, X, M: nonlinear_model(params, X) - M

def fit_nonlinear_single(X, M):
    """单角度非线性拟合"""
    slope, intercept = np.polyfit(X, M, 1)
    result = least_squares(residuals, [slope, intercept, 0.0], args=(X, M))
    d, b, c = result.x
    M_pred = nonlinear_model(result.x, X)
    sse = np.sum((M - M_pred)**2)
    n = len(M)
    r2 = 1 - sse/np.sum((M - M.mean())**2)
    aic = n*log(sse/n) + 6
    jac = result.jac
    cov = np.linalg.inv(jac.T @ jac) * (sse / (n - 3))
    se = np.sqrt(np.diag(cov))
    return {'d_um': d*1e4, 'd_se_um': se[0]*1e4, 'c': c, 'c_se': se[2], 'r2': r2, 'aic': aic}

def joint_nonlinear_fit(X1, M1, X2, M2):
    """联合非线性拟合"""
    joint_model = lambda params, X1, X2: np.concatenate([
        params[0]*X1 + params[1] + params[3]*X1**2,
        params[0]*X2 + params[2] + params[3]*X2**2
    ])
    joint_residuals = lambda params: joint_model(params, X1, X2) - np.concatenate([M1, M2])
    
    slope1, int1 = np.polyfit(X1, M1, 1)
    slope2, int2 = np.polyfit(X2, M2, 1)
    result = least_squares(joint_residuals, [0.5*(slope1+slope2), int1, int2, 0.0])
    d, b1, b2, c = result.x
    
    M_pred = joint_model(result.x, X1, X2)
    M_obs = np.concatenate([M1, M2])
    sse = np.sum((M_obs - M_pred)**2)
    n = len(M_obs)
    aic = n*log(sse/n) + 8
    jac = result.jac
    cov = np.linalg.inv(jac.T @ jac) * (sse / (n - 4))
    se = np.sqrt(np.diag(cov))
    
    return {'d_um': d*1e4, 'd_se_um': se[0]*1e4, 'c': c, 'c_se': se[3], 'aic': aic, 'n': n}

def analyze_nonlinear(file1_path, theta1, file2_path=None, theta2=None, prominence=0.002, distance=10):
    """非线性最小二乘分析"""
    models = {'4H': n2_4H, '6H': n2_6H}
    results = {}
    
    print("非线性拟合分析 (二次项修正)")
    
    for name, n2_fn in models.items():
        print(f'\n{name}-SiC模型 (M=dX+b+cX²):')
        
        nu1, R1 = read_spectrum(file1_path)
        peaks1 = extract_peaks(nu1, R1, prominence, distance)
        X1, _ = compute_X(peaks1, theta1, n2_fn)
        M1 = np.arange(1, len(X1)+1, dtype=float)
        
        res1 = fit_nonlinear_single(X1, M1)
        print(f"附件1({theta1}°): d={res1['d_um']:.3f}±{1.96*res1['d_se_um']:.3f}μm, c={res1['c']:.2e}, R²={res1['r2']:.3f}")
        
        if file2_path and theta2:
            nu2, R2 = read_spectrum(file2_path)
            peaks2 = extract_peaks(nu2, R2, prominence, distance)
            X2, _ = compute_X(peaks2, theta2, n2_fn)
            M2 = np.arange(1, len(X2)+1, dtype=float)
            
            res2 = fit_nonlinear_single(X2, M2)
            print(f"附件2({theta2}°): d={res2['d_um']:.3f}±{1.96*res2['d_se_um']:.3f}μm, c={res2['c']:.2e}, R²={res2['r2']:.3f}")
            
            joint_res = joint_nonlinear_fit(X1, M1, X2, M2)
            print(f"联合拟合: d={joint_res['d_um']:.3f}±{1.96*joint_res['d_se_um']:.3f}μm")
            print(f"          共享c={joint_res['c']:.2e}±{1.96*joint_res['c_se']:.2e}, AIC={joint_res['aic']:.1f}")
            
            # 一致性评估
            d1, d2 = res1['d_um'], res2['d_um']
            consistency_error = abs(d1 - d2) / (0.5*(d1 + d2)) * 100
            consistency = "好" if consistency_error < 5 else "一般" if consistency_error < 15 else "差"
            print(f"一致性: {consistency} (偏差{consistency_error:.1f}%)")
            
            results[name] = {'single_1': res1, 'single_2': res2, 'joint': joint_res, 'consistency': consistency}
        else:
            results[name] = {'single_1': res1}
    
    return results

if __name__ == '__main__':
    results = analyze_nonlinear("附件/附件1.xlsx", 10, "附件/附件2.xlsx", 15)
    
    print("\n最终结果:")
    for model in ['4H', '6H']:
        if model in results and 'joint' in results[model]:
            joint = results[model]['joint']
            print(f"{model}: {joint['d_um']:.3f}±{1.96*joint['d_se_um']:.3f}μm, 非线性c={joint['c']:.1e}")
