import numpy as np
from scipy.optimize import differential_evolution
from math import sin, radians, pi, sqrt
import pandas as pd
from scipy.signal import find_peaks
from Q2_LR import read_spectrum, n2_4H, n2_6H, compute_X, linfit_X_M, extract_peaks
import warnings
warnings.filterwarnings('ignore')


def n_SiC(lam_um, sic_type='4H'):
    if sic_type == '4H':
        return np.sqrt(n2_4H(lam_um))
    else:
        return np.sqrt(n2_6H(lam_um))

def fabry_perot_model_sic(nu_cm, thickness_um, theta_deg=0, absorption_coeff=0.001, sic_type='4H'):
    lambda_um = 1e4 / nu_cm
    n_epi = n_SiC(lambda_um, sic_type)
    n_sub = n_epi * 1.001
    
    theta_i, theta_t = radians(theta_deg), np.arcsin(np.clip(sin(radians(theta_deg)) / n_epi, -1, 1))
    cos_theta_i, cos_theta_t = np.cos(theta_i), np.cos(theta_t)
    
    # 界面1: 空气->SiC外延层
    r_s1 = (cos_theta_i - n_epi * cos_theta_t) / (cos_theta_i + n_epi * cos_theta_t)
    r_p1 = (n_epi * cos_theta_i - cos_theta_t) / (n_epi * cos_theta_i + cos_theta_t)
    R1 = 0.5 * (r_s1**2 + r_p1**2)
    
    # 界面2: SiC外延层->SiC衬底
    cos_theta_sub = np.cos(np.arcsin(np.clip(n_epi * np.sin(theta_t) / n_sub, -1, 1)))
    r_s2 = (n_epi * cos_theta_t - n_sub * cos_theta_sub) / (n_epi * cos_theta_t + n_sub * cos_theta_sub)
    r_p2 = (n_sub * cos_theta_t - n_epi * cos_theta_sub) / (n_sub * cos_theta_t + n_epi * cos_theta_sub)
    R2 = 0.5 * (r_s2**2 + r_p2**2)
    
    R2_prime = R2 * np.exp(-2 * (4 * pi * absorption_coeff / lambda_um) * thickness_um / cos_theta_t)
    delta = 4 * pi * thickness_um * n_epi * cos_theta_t / lambda_um
    sqrt_R1R2_prime = np.sqrt(R1 * R2_prime)
    
    return (R1 + R2_prime - 2 * sqrt_R1R2_prime * np.cos(delta)) / (1 + R1 * R2_prime - 2 * sqrt_R1R2_prime * np.cos(delta))

extract_peaks_sic = lambda nu, R: extract_peaks(nu, R, prominence=0.003, distance=15)

def two_beam_linear_fit(nu_peaks, theta_deg, sic_type='4H'):
    n2_fn = n2_4H if sic_type == '4H' else n2_6H
    X, mask = compute_X(nu_peaks, theta_deg, n2_fn)
    M = np.arange(1, len(X) + 1, dtype=float)
    d_cm, b, r2, r2_adj, sse, aic, se_d_cm, n = linfit_X_M(X, M)
    
    return {'thickness_um': d_cm * 1e4, 'thickness_se_um': se_d_cm * 1e4, 'r2': r2, 'n_peaks': len(X), 'baseline': b}

def fit_fabry_perot_sic(file_path, theta_deg, sic_type='4H', two_beam_initial=None):
    nu, R = read_spectrum(file_path)
    nu_peaks = extract_peaks_sic(nu, R)
    
    if two_beam_initial is None:
        initial_thickness = two_beam_linear_fit(nu_peaks, theta_deg, sic_type)['thickness_um']
    else:
        initial_thickness = two_beam_initial
    
    def objective_function(params):
        thickness_um, amplitude, baseline, absorption_coeff = params
        if (thickness_um <= 0 or thickness_um > 15 or amplitude <= 0 or amplitude > 2.0 or
            baseline < 0 or baseline > 1.0 or absorption_coeff < 0 or absorption_coeff > 0.01):
            return 1e8
        

        R_theory = fabry_perot_model_sic(nu, thickness_um, theta_deg, absorption_coeff, sic_type)
        if np.any(np.isnan(R_theory)) or np.any(np.isinf(R_theory)):
            return 1e8
        
        R_model = amplitude * R_theory + baseline
        mse = np.mean((R - R_model)**2)
        thickness_penalty = 0.001 * ((thickness_um - initial_thickness) / initial_thickness)**2
        return mse + thickness_penalty

    
    bounds = [(max(0.5, initial_thickness * 0.8), min(15.0, initial_thickness * 1.2)), (0.05, 2.0), (0.0, 1.0), (0.0001, 0.01)]
    
    result = differential_evolution(objective_function, bounds, seed=42, maxiter=2000, atol=1e-8, tol=1e-6)
    
    if result.success:
        thickness_opt, amp_opt, bg_opt, abs_opt = result.x
        R_theory = fabry_perot_model_sic(nu, thickness_opt, theta_deg, abs_opt, sic_type)
        R_model = amp_opt * R_theory + bg_opt
        r2 = 1 - np.sum((R - R_model) ** 2) / np.sum((R - np.mean(R)) ** 2)
        
        return {'thickness_um': thickness_opt, 'amplitude': amp_opt, 'baseline': bg_opt, 
                'absorption': abs_opt, 'r2': r2, 'success': True, 'fit_data': (nu, R, R_model)}
    else:
        return {'thickness_um': initial_thickness, 'success': False}

def analyze_single_file(file_path, theta, sic_type):
    nu, R = read_spectrum(file_path)
    nu_peaks = extract_peaks_sic(nu, R)
    two_beam = two_beam_linear_fit(nu_peaks, theta, sic_type)
    
    fp = fit_fabry_perot_sic(file_path, theta, sic_type, two_beam['thickness_um'])
    if fp['success']:
        absolute_effect = fp['thickness_um'] - two_beam['thickness_um']
        relative_effect = absolute_effect / two_beam['thickness_um'] * 100
        print(f"θ={theta}°: 双光束 {two_beam['thickness_um']:.3f}μm → F-P {fp['thickness_um']:.3f}μm (Δ{relative_effect:+.1f}%)")
    else:
        print(f"θ={theta}°: 双光束 {two_beam['thickness_um']:.3f}μm (F-P拟合失败)")
    
    return {'two_beam': two_beam, 'fp': fp}

def analyze_multibeam_interference_effect(file1_path, theta1, file2_path=None, theta2=None):
    print("SiC外延层多光束干涉效应分析")
    
    results = {}
    
    for sic_type in ['4H', '6H']:
        print(f"\n{sic_type}-SiC:")
        
        file1_result = analyze_single_file(file1_path, theta1, sic_type)
        sic_results = {'file1': file1_result}
        
        if file2_path and theta2:
            file2_result = analyze_single_file(file2_path, theta2, sic_type)
            sic_results['file2'] = file2_result
            
            fp_1, fp_2 = file1_result['fp'], file2_result['fp']
            two_beam_1, two_beam_2 = file1_result['two_beam'], file2_result['two_beam']
            
            if fp_1['success'] and fp_2['success']:
                two_beam_joint = (two_beam_1['thickness_um'] + two_beam_2['thickness_um']) / 2
                fp_joint = (fp_1['thickness_um'] + fp_2['thickness_um']) / 2
                
                joint_effect = (fp_joint - two_beam_joint) / two_beam_joint * 100
                print(f"联合结果: 双光束 {two_beam_joint:.3f}μm → F-P {fp_joint:.3f}μm (系统误差{joint_effect:+.1f}%)")
                
                sic_results['joint'] = {
                    'two_beam_joint': two_beam_joint, 'fp_joint': fp_joint,
                    'relative_effect': joint_effect
                }
        
        results[sic_type] = sic_results
    
    return results

def generate_final_report(results):
    print("\n分析结果:")
    
    for sic_type in ['4H', '6H']:
        if sic_type in results and 'joint' in results[sic_type]:
            joint_data = results[sic_type]['joint']
            effect = joint_data['relative_effect']
            
            print(f"{sic_type}: 双光束 {joint_data['two_beam_joint']:.3f}μm → F-P {joint_data['fp_joint']:.3f}μm")
            print(f"系统误差 {effect:+.1f}%, {'需要修正' if abs(effect) > 0.5 else '影响较小'}")

if __name__ == '__main__':
    results = analyze_multibeam_interference_effect("附件/附件1.xlsx", 10, "附件/附件2.xlsx", 15)
    generate_final_report(results)
