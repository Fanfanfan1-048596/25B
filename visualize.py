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

if __name__ == '__main__':
    plot_spectrum_analysis()
    print("光谱分析图已保存")
