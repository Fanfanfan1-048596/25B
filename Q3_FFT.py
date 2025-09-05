import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import quad
from math import sin, radians, sqrt
import warnings
warnings.filterwarnings('ignore')

def read_spectrum_si(path):
    df = pd.read_excel(path)
    nu_vals = pd.to_numeric(df['波数 (cm-1)'], errors='coerce').to_numpy()
    R_vals = pd.to_numeric(df['反射率 (%)'], errors='coerce').to_numpy()
    mask = ~(np.isnan(nu_vals) | np.isnan(R_vals))
    nu, R = nu_vals[mask], R_vals[mask]
    order = np.argsort(nu)
    return nu[order], R[order] / 100.0

def n_si(lam_um):
    """硅的Sellmeier方程"""
    A1, B1 = 10.6684293, 0.301516485
    A2, B2 = 0.0030434748, 1.13475115
    n_squared = 1.0 + (A1 * lam_um**2) / (lam_um**2 - B1**2) + (A2 * lam_um**2) / (lam_um**2 - B2**2)
    return np.sqrt(n_squared)

def compute_neff_numerical(nu_min, nu_max):
    """数值积分计算有效折射率"""
    def integrand(nu):
        lam = 1e4 / nu  # cm^-1 转换为 μm
        return n_si(lam)
    
    integral, _ = quad(integrand, nu_min, nu_max)
    n_eff = integral / (nu_max - nu_min)
    return n_eff

def preprocess_spectrum(nu, R):
    R_smooth = savgol_filter(R, window_length=51, polyorder=3)
    background = savgol_filter(R_smooth, window_length=501, polyorder=2)
    return R_smooth, R_smooth - background, background

def fft_thickness_analysis(nu, R_osc, theta_deg=0):
    # 定义每个文件对应的极小值范围
    minima_ranges = {
        10: (568.9, 3851.1),  # 附件3: θ=10°，从极小值1到极小值9
        15: (574.2, 3910.0)   # 附件4: θ=15°，从极小值1到极小值9
    }
    
    # 获取对应角度的分析范围
    if theta_deg in minima_ranges:
        nu_min, nu_max = minima_ranges[theta_deg]
        print(f"使用极小值范围: [{nu_min:.1f}, {nu_max:.1f}] cm^-1")
    
    # 截取范围内数据并重采样
    range_mask = (nu >= nu_min) & (nu <= nu_max)
    nu_range, R_range = nu[range_mask], R_osc[range_mask]
    
    if len(nu_range) < 100:
        return 0, 0, np.array([]), np.array([]), [], 0
    
    # FFT分析
    N = 2**14
    nu_uniform = np.linspace(nu_min, nu_max, N)
    R_uniform = np.interp(nu_uniform, nu_range, R_range)
    fft_vals = fft(R_uniform)
    
    # 构建长度轴
    d_nu = nu_uniform[1] - nu_uniform[0]
    l_axis = fftfreq(N, d=d_nu)
    pos_mask = l_axis > 0
    l_pos, fft_pos = l_axis[pos_mask], np.abs(fft_vals[pos_mask])
    
    # 寻找主峰
    peak_idx = find_peaks(fft_pos, height=np.max(fft_pos)*0.1)[0]
    if len(peak_idx) == 0:
        return 0, 0, l_pos, fft_pos, [], 0
    
    main_peak = peak_idx[np.argmax(fft_pos[peak_idx])]
    opd_cm = l_pos[main_peak]
    
    # 计算厚度
    n_eff = compute_neff_numerical(nu_min, nu_max)  # 使用数值积分平均法
    
    sin_theta = sin(radians(theta_deg))
    thickness_cm = opd_cm / (2 * sqrt(n_eff**2 - sin_theta**2))
    thickness_um = thickness_cm * 1e4
    
    return thickness_um, opd_cm, l_pos, fft_pos, peak_idx, n_eff

def analyze_interference_condition(nu, R, theta_deg=0):
    wavelength = 1e4 / nu
    n_silicon = n_si(wavelength)
    
    cos_theta = np.cos(radians(theta_deg))
    r = (cos_theta - n_silicon*np.sqrt(1-(sin(radians(theta_deg))/n_silicon)**2)) / \
        (cos_theta + n_silicon*np.sqrt(1-(sin(radians(theta_deg))/n_silicon)**2))
    
    avg_reflectance = np.mean(np.abs(r)**2)
    R_smooth, R_osc, _ = preprocess_spectrum(nu, R)
    osc_amplitude = np.std(R_osc)
    
    return {
        'high_reflectance': avg_reflectance > 0.3,
        'regular_oscillation': osc_amplitude > 0.01,
        'avg_reflectance': avg_reflectance,
        'oscillation_amplitude': osc_amplitude,
        'suitable_for_multibeam': avg_reflectance > 0.3 and osc_amplitude > 0.01
    }

def analyze_si_wafer_fft(file_path, theta_deg=0):
    nu, R = read_spectrum_si(file_path)
    
    print(f"{file_path} FFT分析")
    print(f"数据点数: {len(nu)}, 波数: {nu.min():.1f}-{nu.max():.1f} cm^-1")
    
    R_smooth, R_osc, background = preprocess_spectrum(nu, R)
    
    conditions = analyze_interference_condition(nu, R, theta_deg)
    print(f"反射率: {conditions['avg_reflectance']:.3f}, 振荡: {conditions['oscillation_amplitude']:.4f}")
    print(f"多光束干涉: {'适用' if conditions['suitable_for_multibeam'] else '不适用'}")
    
    thickness_um, opd_cm, l_axis, fft_vals, peak_indices, n_eff = fft_thickness_analysis(nu, R_osc, theta_deg)
    
    print(f"厚度: {thickness_um:.3f} μm (n_eff={n_eff:.3f}, OPD={opd_cm:.6f} cm)")
    
    return {
        'thickness_um': thickness_um,
        'opd_cm': opd_cm,
        'n_effective': n_eff,
        'interference_conditions': conditions,
        'spectrum_data': (nu, R, R_smooth, R_osc, background),
        'fft_data': (l_axis, fft_vals)
    }

if __name__ == '__main__':
    files = [('附件/附件3.xlsx', 10), ('附件/附件4.xlsx', 15)]
    results = {}
    
    for file_path, theta in files:
        result = analyze_si_wafer_fft(file_path, theta)
        results[file_path] = result
        print()
    
    print("最终结果")
    for file_path, result in results.items():
        filename = file_path.split('/')[-1]
        print(f"{filename}: {result['thickness_um']:.3f} μm")
