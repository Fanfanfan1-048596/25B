import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from math import sin, radians, sqrt, log

def read_spectrum(path):
    df = pd.read_excel(path)
    nu_vals = pd.to_numeric(df['波数 (cm-1)'], errors='coerce').to_numpy()
    R_vals = pd.to_numeric(df['反射率 (%)'], errors='coerce').to_numpy()
    mask = ~(np.isnan(nu_vals) | np.isnan(R_vals))
    nu, R = nu_vals[mask], R_vals[mask]
    
    # 对附件1和附件2，只使用波数>1500的数据
    if '附件1' in path or '附件2' in path:
        wavenumber_mask = nu > 2000
        nu, R = nu[wavenumber_mask], R[wavenumber_mask]
    
    order = np.argsort(nu)
    return nu[order], R[order] / 100.0

# 4H/6H-SiC折射率模型
n2_4H = lambda lam: 1.0 + 5.54861*lam**2/(lam**2 - 0.02641) + 0.20075*lam**2/(lam**2 + 12.07224) + 35.65066*lam**2/(lam**2 - 1268.24708)
n2_6H = lambda lam: 6.57232 + 0.1401/(lam**2 - 0.03178) - 0.02153*lam**2

def compute_X(nu_cm, theta_deg, n2_fn):
    lam_um = 1e4 / nu_cm
    arg = n2_fn(lam_um) - sin(radians(theta_deg))**2
    mask = arg > 0
    return 2.0 * nu_cm[mask] * np.sqrt(arg[mask]), mask

def linfit_X_M(X, M):
    X, M = np.asarray(X, float), np.asarray(M, float)
    n = len(M)
    A = np.column_stack([X, np.ones_like(X)])
    d_cm, b = np.linalg.lstsq(A, M, rcond=None)[0]
    resid = M - (A @ [d_cm, b])
    sse = np.dot(resid, resid)
    tss = np.dot(M - M.mean(), M - M.mean())
    r2 = 1.0 - sse/tss if tss > 0 else np.nan
    r2_adj = 1.0 - (sse/(n-2)) / (tss/(n-1)) if n > 2 and tss > 0 else np.nan
    se_d_cm = sqrt((sse/(n-2)) * np.linalg.inv(A.T @ A)[0,0])
    aic = n*log(sse/n) + 4 if sse > 0 else -np.inf
    return d_cm, b, r2, r2_adj, sse, aic, se_d_cm, n

def joint_fit_two_angles(X1, M1, X2, M2):
    X1, M1, X2, M2 = map(lambda x: np.asarray(x, float), [X1, M1, X2, M2])
    n1, n2 = len(M1), len(M2)
    X = np.concatenate([X1, X2])
    M = np.concatenate([M1, M2])
    A = np.column_stack([X, np.concatenate([np.ones(n1), np.zeros(n2)]), np.concatenate([np.zeros(n1), np.ones(n2)])])
    d_cm, b1, b2 = np.linalg.lstsq(A, M, rcond=None)[0]
    sse = np.dot(M - (A @ [d_cm, b1, b2]), M - (A @ [d_cm, b1, b2]))
    se_d_cm = sqrt((sse/(len(M)-3)) * np.linalg.inv(A.T @ A)[0,0])
    aic = len(M)*log(sse/len(M)) + 6 if sse > 0 else -np.inf
    return d_cm, b1, b2, sse, aic, se_d_cm, len(M), 3

def filter_peaks_by_distance(peaks, min_distance=200):
    """根据最小距离过滤峰值"""
    if len(peaks) <= 1:
        return peaks
    
    filtered_peaks = [peaks[0]]  # 保留第一个峰值
    
    for peak in peaks[1:]:
        if peak - filtered_peaks[-1] >= min_distance:
            filtered_peaks.append(peak)
    
    return np.array(filtered_peaks)

extract_peaks = lambda nu, R, prominence=0.002, distance=10: filter_peaks_by_distance(np.sort(nu[find_peaks(R, prominence=prominence, distance=distance)[0]]), min_distance=200)

def fit_one_angle(path, theta_deg, n2_fn, prominence, distance):
    nu, R = read_spectrum(path)
    nu_pk = extract_peaks(nu, R, prominence, distance)
    X, mask = compute_X(nu_pk, theta_deg, n2_fn)
    M = np.arange(1, len(X)+1, dtype=float)
    d_cm, b, r2, r2_adj, sse, aic, se_d_cm, n = linfit_X_M(X, M)
    return {'n_peaks': len(X), 'd_um': d_cm*1e4, 'd_um_se': se_d_cm*1e4, 'R2': r2, 'R2_adj': r2_adj, 'AIC': aic, 'theta_deg': theta_deg}

def analyze_thickness(file1_path, theta1, file2_path=None, theta2=None, prominence=0.002, distance=10):
    models = {'4H': n2_4H, '6H': n2_6H}
    results = {}
    
    for name, n2_fn in models.items():
        print(f'{name}-SiC线性拟合:')
        res1 = fit_one_angle(file1_path, theta1, n2_fn, prominence, distance)
        print(f"附件1({theta1}°): d={res1['d_um']:.3f}±{1.96*res1['d_um_se']:.3f}μm, R^2={res1['R2_adj']:.3f}, {res1['n_peaks']}个峰")

        if file2_path and theta2:
            res2 = fit_one_angle(file2_path, theta2, n2_fn, prominence, distance)
            print(f"附件2({theta2}°): d={res2['d_um']:.3f}±{1.96*res2['d_um_se']:.3f}μm, R^2={res2['R2_adj']:.3f}, {res2['n_peaks']}个峰")

            # 联合回归
            nu1, R1 = read_spectrum(file1_path)
            nu2, R2 = read_spectrum(file2_path)
            X1, _ = compute_X(extract_peaks(nu1, R1, prominence, distance), theta1, n2_fn)
            X2, _ = compute_X(extract_peaks(nu2, R2, prominence, distance), theta2, n2_fn)
            M1, M2 = np.arange(1, len(X1)+1, dtype=float), np.arange(1, len(X2)+1, dtype=float)
            d_cm, b1, b2, sse, aic, se_d_cm, n, p = joint_fit_two_angles(X1, M1, X2, M2)
            print(f"联合回归: d={d_cm*1e4:.3f}±{1.96*se_d_cm*1e4:.3f}μm, AIC={aic:.1f}")
            
            d1, d2 = res1['d_um'], res2['d_um']
            rel = abs(d1 - d2) / (0.5*(d1 + d2)) * 100
            consistency = "好" if rel < 5 else "一般" if rel < 15 else "差"
            print(f"一致性: {consistency} (偏差{rel:.1f}%)")
            
            results[name] = {'single_angle_1': res1, 'single_angle_2': res2, 'joint_thickness': d_cm*1e4, 'joint_se': se_d_cm*1e4, 'joint_aic': aic, 'consistency_error': rel}
        else:
            results[name] = {'single_angle_1': res1}
    
    return results

if __name__ == '__main__':
    results = analyze_thickness("附件/附件1.xlsx", 10, "附件/附件2.xlsx", 15)
    print("\n最终结果:")
    for model in ['4H', '6H']:
        if model in results and 'joint_aic' in results[model]:
            r = results[model]
            print(f"{model}: {r['joint_thickness']:.3f}±{1.96*r['joint_se']:.3f}μm, AIC={r['joint_aic']:.1f}")