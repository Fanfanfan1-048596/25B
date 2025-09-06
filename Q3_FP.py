import numpy as np
from scipy.optimize import differential_evolution
from math import sin, radians, pi
from Q3_FFT import read_spectrum_si, n_si, analyze_si_wafer_fft
import warnings
warnings.filterwarnings('ignore')

def fabry_perot_model(nu_cm, thickness_um, theta_deg=0, absorption_coeff=0.01):
    """考虑损耗的法布里-珀罗理论模型"""
    lambda_um = 1e4 / nu_cm
    n_epi = n_si(lambda_um)
    n_sub = n_epi * 1.001
    
    # 计算折射角
    theta_i = radians(theta_deg)
    sin_theta_t = np.clip(sin(theta_i) / n_epi, -1, 1)
    theta_t = np.arcsin(sin_theta_t)
    cos_theta_i, cos_theta_t = np.cos(theta_i), np.cos(theta_t)
    
    # 菲涅尔系数计算
    # 界面1: 空气->外延层
    r_s1 = (cos_theta_i - n_epi * cos_theta_t) / (cos_theta_i + n_epi * cos_theta_t)
    r_p1 = (n_epi * cos_theta_i - cos_theta_t) / (n_epi * cos_theta_i + cos_theta_t)
    R1 = 0.5 * (r_s1**2 + r_p1**2)
    
    # 界面2: 外延层->衬底
    sin_theta_sub = np.clip(n_epi * np.sin(theta_t) / n_sub, -1, 1)
    cos_theta_sub = np.cos(np.arcsin(sin_theta_sub))
    r_s2 = (n_epi * cos_theta_t - n_sub * cos_theta_sub) / (n_epi * cos_theta_t + n_sub * cos_theta_sub)
    r_p2 = (n_sub * cos_theta_t - n_epi * cos_theta_sub) / (n_sub * cos_theta_t + n_epi * cos_theta_sub)
    R2 = 0.5 * (r_s2**2 + r_p2**2)
    
    # 引入损耗项
    alpha = 4 * pi * absorption_coeff / lambda_um
    R2_prime = R2 * np.exp(-2 * alpha * thickness_um / cos_theta_t)
    
    # 艾里反射率公式
    delta = 4 * pi * thickness_um * n_epi * cos_theta_t / lambda_um
    sqrt_R1R2_prime = np.sqrt(R1 * R2_prime)
    cos_delta = np.cos(delta)
    
    return (R1 + R2_prime - 2 * sqrt_R1R2_prime * cos_delta) / (1 + R1 * R2_prime - 2 * sqrt_R1R2_prime * cos_delta)

def fit_fabry_perot(file_path, theta_deg, fft_initial_guess):
    nu, R = read_spectrum_si(file_path)
    
    minima_ranges = {10: (568.9, 3851.1), 15: (574.2, 3910.0)}
    nu_min, nu_max = minima_ranges[theta_deg]
    print(f"FP拟合范围:[{nu_min:.1f}, {nu_max:.1f}]")
    
    range_mask = (nu >= nu_min) & (nu <= nu_max)
    nu_fit, R_fit = nu[range_mask], R[range_mask]
    print(f"拟合数据点数: {len(nu_fit)}")
    
    def objective_function(params):
        thickness_um, amplitude_factor, background_level, absorption_coeff = params
        
        if (thickness_um <= 0 or thickness_um > 20 or amplitude_factor <= 0 or amplitude_factor > 3.0 or
            background_level < 0 or background_level > 0.5 or absorption_coeff < 0 or absorption_coeff > 0.05):
            return 1e8
        
        try:
            R_theory = fabry_perot_model(nu_fit, thickness_um, theta_deg, absorption_coeff)
            if np.any(np.isnan(R_theory)) or np.any(np.isinf(R_theory)):
                return 1e8
            
            R_model_scaled = amplitude_factor * R_theory + background_level
            residual = R_fit - R_model_scaled
            mse = np.mean(residual**2)
            thickness_penalty = 0.001 * ((thickness_um - fft_initial_guess) / fft_initial_guess)**2
            
            return mse + thickness_penalty
        except Exception:
            return 1e8

    # 参数设置
    bounds = [(6.0, 12.0), (0.05, 3.0), (0.0, 0.5), (0.0001, 0.05)]
    R_mean, R_std = np.mean(R_fit), np.std(R_fit)
    initial_guess = [fft_initial_guess, min(2.0 * R_std, 1.5), max(R_mean - R_std, 0.1), 0.005]
    
    result = differential_evolution(objective_function, bounds, seed=42, maxiter=3000, atol=1e-8, tol=1e-6)
    
    if result.success:
        thickness_opt, amp_opt, bg_opt, abs_opt = result.x
        R_theory = fabry_perot_model(nu_fit, thickness_opt, theta_deg, abs_opt)
        R_model = amp_opt * R_theory + bg_opt
        
        return {'thickness_um': thickness_opt, 'background': bg_opt, 'amplitude': amp_opt,
                'absorption': abs_opt, 'success': True, 'fit_data': (nu_fit, R_fit, R_model)}
    else:
        return {'thickness_um': fft_initial_guess, 'background': 0, 'amplitude': 0,
                'absorption': 0, 'success': False, 'fit_data': (nu_fit, R_fit, R_fit)}

def analyze_si_wafer_fabry_perot(file_path, theta_deg, fft_initial_thickness=None):
    print(f"{file_path} FP分析 (theta={theta_deg}°)")
    print(f"提供的初始厚度: {fft_initial_thickness:.3f} μm")
    print("开始非线性拟合...")
    fit_result = fit_fabry_perot(file_path, theta_deg, fft_initial_thickness)
    
    return fit_result

if __name__ == '__main__':
    files_and_angles = [('附件/附件3.xlsx', 10), ('附件/附件4.xlsx', 15)]
    print("硅晶圆片多光束干涉厚度分析\n")
    final_results = {}
    
    for file_path, theta in files_and_angles:
        print("\nFFT频域分析")
        fft_result = analyze_si_wafer_fft(file_path, theta)
        
        print("\nFP多光束干涉拟合")
        fp_result = analyze_si_wafer_fabry_perot(file_path, theta, fft_result['thickness_um'])
        
        if fp_result:
            final_results[file_path] = {
                'fft_thickness': fft_result['thickness_um'],
                'fp_thickness': fp_result['thickness_um'],
                'background': fp_result['background'],
                'amplitude': fp_result['amplitude'],
                'absorption': fp_result['absorption'],
                'interference_suitable': fft_result['interference_conditions']['suitable_for_multibeam']
            }
    
    print("\n最终厚度测量结果")
    for file_path, result in final_results.items():
        filename = file_path.split('/')[-1]
        improvement = abs(result['fp_thickness'] - result['fft_thickness']) / result['fft_thickness'] * 100
        
        print(f"\n{filename}:")
        print(f"FFT初步估计: {result['fft_thickness']:.3f} μm")
        print(f"F-P精确拟合: {result['fp_thickness']:.3f} μm")
        print(f"背景水平: {result['background']:.3f}")
        print(f"幅度因子: {result['amplitude']:.3f}")
        print(f"吸收系数: {result['absorption']:.4f}")
        print(f"多光束干涉适用性: {'是' if result['interference_suitable'] else '否'}")
        print(f"相对FFT变化: {improvement:.1f}%")
