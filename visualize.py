import matplotlib.pyplot as plt
import numpy as np
from Q2_LR import read_spectrum, extract_peaks, n2_4H, n2_6H, compute_X

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_spectrum_analysis():
    # 数据读取
    data = [(read_spectrum("附件/附件1.xlsx"), 10, '附件1: θ=10°'),
            (read_spectrum("附件/附件2.xlsx"), 15, '附件2: θ=15°')]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SiC外延层光谱分析', fontsize=16, fontweight='bold')
    
    # 原始光谱可视化
    for i, ((nu, R), theta, title) in enumerate(data):
        peaks = extract_peaks(nu, R)
        ax = axes[0, i]
        ax.plot(nu, R, ['b-', 'g-'][i], linewidth=1, alpha=0.8, label='原始反射率')
        ax.plot(peaks, R[np.searchsorted(nu, peaks)], 'ro', markersize=4, label=f'检测峰值 ({len(peaks)}个)')
        ax.set_xlabel('波数 (cm^-1)', fontfamily='SimHei')
        ax.set_ylabel('反射率', fontfamily='SimHei')
        ax.set_title(f'{title} 原始光谱与峰值', fontfamily='SimHei')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(nu.min(), nu.max())
    
    # 线性拟合可视化
    for i, (model_name, n2_fn) in enumerate([('4H-SiC模型线性拟合', n2_4H), ('6H-SiC模型线性拟合', n2_6H)]):
        ax = axes[1, i]
        X_all, M_all = [], []
        
        for j, ((nu, R), theta, _) in enumerate(data):
            peaks = extract_peaks(nu, R)
            X, _ = compute_X(peaks, theta, n2_fn)
            M = np.arange(1, len(X)+1)
            ax.scatter(X, M, c=['blue', 'red'][j], s=30, alpha=0.7, label=f'θ={theta}°')
            X_all.extend(X)
            M_all.extend(M)
        
        # 拟合线
        slope, intercept = np.polyfit(X_all, M_all, 1)
        x_line = np.linspace(min(X_all), max(X_all), 100)
        ax.plot(x_line, slope*x_line + intercept, 'k--', alpha=0.8, label=f'拟合线 (斜率={slope*1e4:.3f} μm)')
        
        ax.set_xlabel('X (cm^-1)', fontfamily='SimHei')
        ax.set_ylabel('相对干涉级次 M', fontfamily='SimHei')
        ax.set_title(model_name, fontfamily='SimHei')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('SiC光谱分析.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_additional_spectrum():
    """绘制附件3和附件4的原始光谱与峰值"""
    from scipy.signal import find_peaks
    
    def extract_minima_from_500(nu, R):
        """从波数500开始检索极小值"""
        # 只在波数>500的范围内查找极小值
        mask_500 = nu > 500
        nu_filtered = nu[mask_500]
        R_filtered = R[mask_500]
        
        # 寻找极小值，使用更宽松的参数以确保找到所有重要极小值
        minima_indices = find_peaks(-R_filtered, prominence=0.003, distance=120, height=-0.5)[0]
        minima_nu = nu_filtered[minima_indices]
        
        # 确保选择最显著的9个极小值
        if len(minima_nu) != 9:
            # 按反射率值排序，选择最低的9个点
            minima_R = R_filtered[minima_indices]
            if len(minima_nu) > 9:
                sorted_indices = np.argsort(minima_R)[:9]
            else:
                # 如果少于9个，放宽条件重新搜索
                minima_indices = find_peaks(-R_filtered, prominence=0.002, distance=100)[0]
                minima_nu = nu_filtered[minima_indices]
                minima_R = R_filtered[minima_indices]
                if len(minima_nu) > 9:
                    sorted_indices = np.argsort(minima_R)[:9]
                else:
                    sorted_indices = np.arange(len(minima_nu))
            
            minima_nu = minima_nu[sorted_indices]
            minima_nu = np.sort(minima_nu)  # 按波数重新排序
            
        return minima_nu
    
    # 数据读取
    data = [(read_spectrum("附件/附件3.xlsx"), 10, '附件3: θ=10°'),
            (read_spectrum("附件/附件4.xlsx"), 15, '附件4: θ=15°')]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('硅晶圆片干涉光谱分析 (附件3&4)', fontsize=16, fontweight='bold')
    
    # 原始光谱可视化
    for i, ((nu, R), theta, title) in enumerate(data):
        peaks = extract_peaks(nu, R)
        minima = extract_minima_from_500(nu, R)
        
        ax = axes[i]
        ax.plot(nu, R, ['b-', 'g-'][i], linewidth=1, alpha=0.8, label='原始反射率')
        ax.plot(peaks, R[np.searchsorted(nu, peaks)], 'ro', markersize=4, label=f'极大值 ({len(peaks)}个)')
        ax.plot(minima, R[np.searchsorted(nu, minima)], 'bs', markersize=4, label=f'极小值 ({len(minima)}个)')
        
        ax.set_xlabel('波数 (cm^-1)', fontfamily='SimHei')
        ax.set_ylabel('反射率', fontfamily='SimHei')
        ax.set_title(f'{title} 光谱极值分析', fontfamily='SimHei')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(nu.min(), nu.max())
        
        # 打印极小值坐标
        print(f"\n{title} 极小值坐标 (波数>500):")
        for j, min_nu in enumerate(minima):
            min_R = R[np.searchsorted(nu, min_nu)]
            print(f"  极小值{j+1}: ({min_nu:.1f} cm^-1, {min_R:.4f})")
    
    plt.tight_layout()
    plt.savefig('硅晶圆片光谱分析.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_fft_analysis():
    """绘制附件3和4的FFT分析结果"""
    from Q3_FFT import analyze_si_wafer_fft
    
    files_and_angles = [
        ('附件/附件3.xlsx', 10),
        ('附件/附件4.xlsx', 15)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('硅晶圆片FFT厚度分析', fontfamily='SimHei', fontsize=16, fontweight='bold')
    
    for i, (file_path, theta) in enumerate(files_and_angles):
        result = analyze_si_wafer_fft(file_path, theta)
        nu, R, R_smooth, R_osc, background = result['spectrum_data']
        l_axis, fft_vals = result['fft_data']
        thickness = result['thickness_um']
        opd = result['opd_cm']
        
        # 光谱预处理图
        ax1 = axes[i, 0]
        ax1.plot(nu, R, 'lightgray', alpha=0.7, label='原始光谱')
        ax1.plot(nu, R_smooth, 'b-', label='平滑光谱')
        ax1.plot(nu, background, 'r--', label='背景趋势')
        ax1.set_xlabel('波数 (cm^-1)', fontfamily='SimHei')
        ax1.set_ylabel('反射率', fontfamily='SimHei')
        ax1.set_title(f'附件{i+3} (θ={theta}°) 光谱预处理', fontfamily='SimHei')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # FFT分析图
        ax2 = axes[i, 1]
        # 只显示合理范围内的FFT结果
        valid_range = (l_axis > 0) & (l_axis < 0.02)  # 0-0.02 cm范围
        ax2.plot(l_axis[valid_range]*1e4, fft_vals[valid_range], 'g-', linewidth=1)
        ax2.axvline(opd*1e4, color='red', linestyle='--', 
                   label=f'主峰: OPD={opd*1e4:.1f}μm')
        ax2.set_xlabel('光程差 (μm)', fontfamily='SimHei')
        ax2.set_ylabel('FFT幅度', fontfamily='SimHei')
        ax2.set_title(f'FFT分析: d={thickness:.2f}μm', fontfamily='SimHei')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('附件3_FFT分析.png', dpi=300, bbox_inches='tight')
    plt.savefig('附件4_FFT分析.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_spectrum_analysis()
    print("光谱分析图已保存")
    
    # 绘制附件3和附件4的光谱
    plot_additional_spectrum()
    print("硅晶圆片光谱分析图已保存")
    
    # 绘制FFT分析
    plot_fft_analysis()
    print("FFT分析图已保存")
