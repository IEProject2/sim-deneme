# =============================================================================
# Gelişmiş Input Data Analysis - V2: İstatistiksel Raporlama
# =============================================================================
# Bu kod, improved_data_analysis.py'nin geliştirilmiş versiyonudur.
#
# V2'deki ana iyileştirme:
# - "EXCELLENT/GOOD" gibi kategorik etiketler yerine 3 katmanlı
#   istatistiksel raporlama sistemi:
#   1. Hipotez Testi Sonucu (H₀ kararı)
#   2. AICc Model Karşılaştırma (ΔAICc & Burnham-Anderson kriterleri)
#   3. Doğal Dilde Yorum Cümlesi
#
# Orijinal dosya: improved_data_analysis.py (dokunulmadı)
# =============================================================================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings
import os
import sys
import io
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Windows konsol encoding düzeltmesi
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# =============================================================================
# YAPILANDIRMA
# =============================================================================

@dataclass
class AnalysisConfig:
    """Analiz konfigürasyonu için veri sınıfı"""
    # Dağılım analizi için minimum gözlem sayısı
    min_observations: int = 30
    
    # Test edilecek dağılımlar
    distributions: tuple = (
        stats.norm,        # Normal
        stats.expon,       # Exponential
        stats.gamma,       # Gamma
        stats.weibull_min, # Weibull
        stats.lognorm,     # Log-Normal
    )
    
    # İstatistiksel anlamlılık eşiği
    significance_level: float = 0.05
    
    # Aykırı değer tespit yöntemi: 'iqr' veya 'zscore'
    outlier_method: str = 'iqr'
    
    # Z-score için eşik (outlier_method='zscore' ise)
    zscore_threshold: float = 3.0
    
    # IQR için çarpan (outlier_method='iqr' ise)
    iqr_multiplier: float = 1.5
    
    # Güven aralığı seviyesi
    confidence_level: float = 0.95
    
    # Görselleştirme ayarları
    figure_style: str = "whitegrid"
    color_palette: str = "husl"
    figure_dpi: int = 100
    
    # Antigravity IDE için ek ayarlar
    save_figures: bool = True
    show_figures: bool = False
    output_dir: str = "IDA_Bitirme_V2"
    export_results: bool = True
    # Excel Veri Yükleme Ayarları
    sheet_name: str | int = 0  # 0 ise ilk sayfa yüklenir. Belirli bir sayfa için adını yazın, Örn: 'Üretim_V1'


# =============================================================================
# VERİ YÜKLEYİCİ SINIFI
# =============================================================================

class DataLoader:
    """Excel dosyasından veri yükleme ve ön işleme sınıfı"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.df = None
        self.machines = []
    
    def load_from_file(self, filepath: str) -> pd.DataFrame:
        """Belirtilen dosya yolundan Excel dosyasını yükler"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")
        
        print(f"[*] '{filepath.name}' dosyasi yukleniyor...")
        
        if filepath.suffix.lower() in ['.xlsx', '.xls']:
            sheet_info = f" (Sheet: '{self.config.sheet_name}')" if self.config.sheet_name else ""
            print(f"[*] Reading Excel file{sheet_info}...")
            self.df = pd.read_excel(filepath, sheet_name=self.config.sheet_name)
        elif filepath.suffix.lower() == '.csv':
            self.df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {filepath.suffix}")
        
        print(f"[OK] Dosya basariyla yuklendi! ({len(self.df)} satir, {len(self.df.columns)} sutun)")
        self._parse_machine_data()
        return self.df
    
    def _parse_machine_data(self) -> None:
        """DataFrame'den makine-zaman çiftlerini çıkarır"""
        self.machines = []
        
        for i in range(0, len(self.df.columns), 2):
            if i + 1 >= len(self.df.columns):
                break
            
            raw_name = str(self.df.columns[i]).strip()
            machine_name = re.sub(r'\.\d+$', '', raw_name)
            time_data = self.df.iloc[:, i + 1]
            
            clean_data = pd.to_numeric(time_data, errors='coerce').dropna()
            clean_data = clean_data[clean_data > 0]
            
            if len(clean_data) > 0:
                self.machines.append({
                    'name': machine_name,
                    'data': clean_data.values,
                    'count': len(clean_data)
                })
        
        print(f"[OK] {len(self.machines)} adet makine/islem tespit edildi.")
    
    def get_machine_data(self, machine_name: str) -> Optional[np.ndarray]:
        """Belirtilen makineye ait zaman verisini döndürür"""
        for machine in self.machines:
            if machine['name'] == machine_name:
                return machine['data']
        return None


# =============================================================================
# İSTATİSTİKSEL ANALİZ SINIFI
# =============================================================================

class StatisticalAnalyzer:
    """Betimsel istatistikler ve aykırı değer analizi sınıfı"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def calculate_descriptive_stats(self, data: np.ndarray) -> Dict:
        """Betimsel istatistikleri hesaplar"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        se = std / np.sqrt(n)
        t_value = stats.t.ppf((1 + self.config.confidence_level) / 2, n - 1)
        ci_lower = mean - t_value * se
        ci_upper = mean + t_value * se
        
        return {
            'count': n,
            'mean': mean,
            'std': std,
            'cv_percent': (std / mean) * 100 if mean != 0 else 0,
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'se': se
        }
    
    def detect_outliers(self, data: np.ndarray) -> Dict:
        """Aykırı değerleri tespit eder"""
        if self.config.outlier_method == 'iqr':
            return self._detect_outliers_iqr(data)
        else:
            return self._detect_outliers_zscore(data)
    
    def _detect_outliers_iqr(self, data: np.ndarray) -> Dict:
        """IQR yöntemi ile aykırı değer tespiti"""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.config.iqr_multiplier * iqr
        upper_bound = q3 + self.config.iqr_multiplier * iqr
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outliers = data[outlier_mask]
        
        return {
            'method': 'IQR',
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(outliers),
            'outlier_percent': (len(outliers) / len(data)) * 100,
            'outliers': outliers,
            'outlier_indices': np.where(outlier_mask)[0]
        }
    
    def _detect_outliers_zscore(self, data: np.ndarray) -> Dict:
        """Z-score yöntemi ile aykırı değer tespiti"""
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores > self.config.zscore_threshold
        outliers = data[outlier_mask]
        
        return {
            'method': 'Z-Score',
            'threshold': self.config.zscore_threshold,
            'outlier_count': len(outliers),
            'outlier_percent': (len(outliers) / len(data)) * 100,
            'outliers': outliers,
            'outlier_indices': np.where(outlier_mask)[0]
        }
    
    def test_normality(self, data: np.ndarray) -> Dict:
        """Shapiro-Wilk normallik testi"""
        if len(data) < 3:
            return {'test': 'Shapiro-Wilk', 'statistic': None, 'p_value': None, 'is_normal': None}
        
        if len(data) <= 5000:
            stat, p_value = stats.shapiro(data)
        else:
            sample = np.random.choice(data, 5000, replace=False)
            stat, p_value = stats.shapiro(sample)
        
        is_normal = p_value > self.config.significance_level
        
        return {
            'test': 'Shapiro-Wilk',
            'statistic': stat,
            'p_value': p_value,
            'is_normal': is_normal
        }


# =============================================================================
# DAĞILIM UYDURMA SINIFI (V2 - İstatistiksel Raporlama)
# =============================================================================

# Dağılım adlarının okunabilir karşılıkları
DIST_DISPLAY_NAMES = {
    'norm': 'Normal',
    'expon': 'Exponential',
    'gamma': 'Gamma',
    'weibull_min': 'Weibull',
    'lognorm': 'Log-Normal',
}

# Dağılım parametre adları (okunabilir)
DIST_PARAM_NAMES = {
    'norm': ['loc (μ)', 'scale (σ)'],
    'expon': ['loc', 'scale (λ⁻¹)'],
    'gamma': ['shape (α)', 'loc', 'scale (β)'],
    'weibull_min': ['shape (c)', 'loc', 'scale (λ)'],
    'lognorm': ['shape (σ)', 'loc', 'scale (e^μ)'],
}


class DistributionFitter:
    """Dağılım uydurma ve uyum testi sınıfı - V2
    
    Yenilikler (V1'e göre):
    - Hipotez testi terminolojisi (H₀ kararı)
    - AICc küçük örneklem düzeltmesi
    - ΔAICc model karşılaştırma (Burnham & Anderson, 2002)
    - Doğal dilde yorum cümleleri
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def fit_distributions(self, data: np.ndarray) -> List[Dict]:
        """Tüm dağılımları test eder ve sonuçları döndürür"""
        results = []
        n = len(data)
        
        for dist in self.config.distributions:
            try:
                result = self._fit_single_distribution(data, dist)
                results.append(result)
            except Exception as e:
                continue
        
        # AICc'ye göre sırala (en düşük en iyi)
        results.sort(key=lambda x: x['aicc'])
        
        # ΔAICc hesapla (Burnham & Anderson, 2002)
        if results:
            best_aicc = results[0]['aicc']
            for r in results:
                delta = r['aicc'] - best_aicc
                r['delta_aicc'] = delta
                
                # Burnham & Anderson (2002) model destek kriterleri:
                # ΔAICc 0-2:   Güçlü destek (substantially best)
                # ΔAICc 2-4:   Orta destek (considerably less support)
                # ΔAICc 4-7:   Zayıf destek (much less support)
                # ΔAICc >7:    Çok zayıf destek (essentially no support)
                if delta <= 2:
                    r['model_support'] = "STRONG"
                    r['model_support_tr'] = "Güçlü"
                elif delta <= 4:
                    r['model_support'] = "MODERATE"
                    r['model_support_tr'] = "Orta"
                elif delta <= 7:
                    r['model_support'] = "WEAK"
                    r['model_support_tr'] = "Zayıf"
                else:
                    r['model_support'] = "VERY WEAK"
                    r['model_support_tr'] = "Çok Zayıf"
        
        return results
    
    def _fit_single_distribution(self, data: np.ndarray, dist) -> Dict:
        """Tek bir dağılımı uydurup test eder"""
        n = len(data)
        
        # MLE ile parametre tahmini
        params = dist.fit(data)
        
        # KS Testi
        ks_stat, ks_p = stats.kstest(data, dist.name, args=params)
        
        # Anderson-Darling testi
        ad_result = self._anderson_darling_test(data, dist.name)
        
        # Log-likelihood, AIC, AICc, BIC hesaplama
        k = len(params)  # parametre sayısı
        
        try:
            log_pdf_values = dist.logpdf(data, *params)
            valid_log_pdf = log_pdf_values[np.isfinite(log_pdf_values)]
            
            if len(valid_log_pdf) == 0 or len(valid_log_pdf) < len(data) * 0.9:
                log_likelihood = -np.inf
                aic = np.inf
                aicc = np.inf
                bic = np.inf
            else:
                log_likelihood = np.sum(valid_log_pdf)
                aic = 2 * k - 2 * log_likelihood
                # AICc: küçük örneklem düzeltmesi (Hurvich & Tsai, 1989)
                if n - k - 1 > 0:
                    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
                else:
                    aicc = np.inf
                bic = k * np.log(n) - 2 * log_likelihood
        except:
            log_likelihood = -np.inf
            aic = np.inf
            aicc = np.inf
            bic = np.inf
        
        # ── KATMAN 1: Hipotez Testi Kararı ──
        alpha = self.config.significance_level
        hypothesis_decision = self._make_hypothesis_decision(ks_p, alpha)
        
        return {
            'distribution': dist.name,
            'display_name': DIST_DISPLAY_NAMES.get(dist.name, dist.name),
            'params': params,
            'param_names': DIST_PARAM_NAMES.get(dist.name, [f'param_{i}' for i in range(len(params))]),
            'n_observations': n,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'ad_statistic': ad_result.get('statistic'),
            'ad_critical_values': ad_result.get('critical_values'),
            'log_likelihood': log_likelihood,
            'aic': aic,
            'aicc': aicc,
            'bic': bic,
            'delta_aicc': 0,         # fit_distributions'da hesaplanacak
            'model_support': '',     # fit_distributions'da hesaplanacak
            'model_support_tr': '',  # fit_distributions'da hesaplanacak
            # Hipotez Testi Sonuçları
            'hypothesis_decision': hypothesis_decision,
            'alpha_used': alpha,
            'is_acceptable': ks_p > alpha and np.isfinite(aicc),
        }
    
    def _make_hypothesis_decision(self, p_value: float, alpha: float) -> Dict:
        """KS testi için hipotez testi kararını oluşturur.
        
        H₀: Veri, belirtilen dağılımdan gelir
        H₁: Veri, belirtilen dağılımdan gelmez
        
        Karar: p > α ise H₀ reddedilemez, aksi halde H₀ reddedilir.
        
        Returns:
            Dict with:
                - rejected: bool
                - label: str (NOT REJECTED / REJECTED)
                - label_tr: str (REDDEDİLEMEDİ / REDDEDİLDİ)
                - confidence_note: str (p-değerinin α'ya yakınlığına göre not)
        """
        rejected = p_value <= alpha
        
        if rejected:
            label = "REJECTED"
            label_tr = "REDDEDİLDİ"
            if p_value < 0.001:
                confidence_note = "Çok güçlü kanıt (p < 0.001)"
                confidence_note_en = "Very strong evidence against H₀ (p < 0.001)"
            elif p_value < 0.01:
                confidence_note = "Güçlü kanıt (p < 0.01)"
                confidence_note_en = "Strong evidence against H₀ (p < 0.01)"
            else:
                confidence_note = "Yeterli kanıt (p ≤ α)"
                confidence_note_en = "Sufficient evidence against H₀ (p ≤ α)"
        else:
            label = "NOT REJECTED"
            label_tr = "REDDEDİLEMEDİ"
            if p_value > 0.10:
                confidence_note = "Güçlü uyum (p > 0.10)"
                confidence_note_en = "Strong agreement with H₀ (p > 0.10)"
            elif p_value > alpha:
                confidence_note = "Sınırda uyum (α < p ≤ 0.10)"
                confidence_note_en = "Marginal agreement (α < p ≤ 0.10)"
            else:
                confidence_note = "Uyum yok"
                confidence_note_en = "No agreement"
        
        return {
            'rejected': rejected,
            'label': label,
            'label_tr': label_tr,
            'confidence_note': confidence_note,
            'confidence_note_en': confidence_note_en,
            'p_value': p_value,
            'alpha': alpha,
        }
    
    def _anderson_darling_test(self, data: np.ndarray, dist_name: str) -> Dict:
        """Anderson-Darling testi (desteklenen dağılımlar için)"""
        supported_dists = {'norm': 'norm', 'expon': 'expon'}
        
        if dist_name in supported_dists:
            try:
                result = stats.anderson(data, dist=supported_dists[dist_name])
                return {
                    'statistic': result.statistic,
                    'critical_values': dict(zip(result.significance_level, result.critical_values))
                }
            except:
                pass
        
        return {'statistic': None, 'critical_values': None}
    
    def get_best_fit(self, results: List[Dict]) -> Optional[Dict]:
        """En iyi dağılımı döndürür (en düşük AICc)"""
        if not results:
            return None
        return results[0]
    
    def get_tied_candidates(self, results: List[Dict], threshold: float = 2.0) -> List[Dict]:
        """ΔAICc ≤ threshold olan tüm modelleri döndürür (istatistiksel olarak eşdeğer)"""
        if not results:
            return []
        return [r for r in results if r['delta_aicc'] <= threshold]
    
    def generate_interpretation(self, machine_name: str, best: Dict, 
                                 desc_stats: Dict, tied_count: int) -> Dict:
        """Her makine için doğal dilde yorum cümlesi üretir.
        
        Returns:
            Dict with 'en' (English) and 'tr' (Turkish) interpretations
        """
        dist_name = best['display_name']
        n = best['n_observations']
        ks_d = best['ks_statistic']
        ks_p = best['ks_p_value']
        aicc = best['aicc']
        alpha = best['alpha_used']
        h0_decision = best['hypothesis_decision']
        
        # Parametre string'i oluştur
        param_pairs = []
        for pname, pval in zip(best['param_names'], best['params']):
            param_pairs.append(f"{pname}={pval:.4f}")
        params_str = ", ".join(param_pairs)
        
        # ── Türkçe Yorum ──
        if h0_decision['rejected']:
            decision_tr = (
                f"α={alpha} anlamlılık düzeyinde H₀ hipotezi reddedilmiştir "
                f"(p={ks_p:.4f} ≤ α={alpha}). "
                f"Bu dağılım veriye istatistiksel olarak uygun bulunamamıştır."
            )
        else:
            decision_tr = (
                f"α={alpha} anlamlılık düzeyinde H₀ hipotezi reddedilememiştir "
                f"(p={ks_p:.4f} > α={alpha}). "
                f"{h0_decision['confidence_note']}."
            )
        
        tied_note_tr = ""
        if tied_count > 1:
            tied_note_tr = (
                f" Not: ΔAICc ≤ 2 kriterine göre {tied_count} model istatistiksel olarak "
                f"birbirine yakın performans göstermiştir; bu durumda en basit model (en az "
                f"parametreli) tercih edilebilir."
            )
        
        interpretation_tr = (
            f"{machine_name} için toplanan {n} gözlem üzerinde yapılan dağılım analizi sonucunda, "
            f"Kolmogorov-Smirnov testi (D={ks_d:.4f}, p={ks_p:.4f}) ve AICc kriteri ({aicc:.2f}) "
            f"baz alınarak {dist_name} dağılımı en uygun model olarak seçilmiştir. "
            f"{decision_tr}"
            f" Tahmini parametreler: {params_str}."
            f"{tied_note_tr}"
        )
        
        # ── English Interpretation ──
        if h0_decision['rejected']:
            decision_en = (
                f"At α={alpha}, H₀ was rejected (p={ks_p:.4f} ≤ α={alpha}). "
                f"This distribution does not statistically fit the data."
            )
        else:
            decision_en = (
                f"At α={alpha}, H₀ could not be rejected (p={ks_p:.4f} > α={alpha}). "
                f"{h0_decision['confidence_note_en']}."
            )
        
        tied_note_en = ""
        if tied_count > 1:
            tied_note_en = (
                f" Note: {tied_count} models are statistically tied (ΔAICc ≤ 2); "
                f"the simplest model (fewest parameters) may be preferred."
            )
        
        interpretation_en = (
            f"Distribution analysis on {n} observations for {machine_name}: "
            f"Based on the Kolmogorov-Smirnov test (D={ks_d:.4f}, p={ks_p:.4f}) and "
            f"AICc criterion ({aicc:.2f}), {dist_name} was selected as the best-fitting model. "
            f"{decision_en}"
            f" Estimated parameters: {params_str}."
            f"{tied_note_en}"
        )
        
        return {
            'tr': interpretation_tr,
            'en': interpretation_en,
        }


# =============================================================================
# GÖRSELLEŞTİRME SINIFI (V2)
# =============================================================================

class Visualizer:
    """Görselleştirme sınıfı - V2: İstatistiksel bilgi ağırlıklı"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) if config.output_dir else Path.cwd()
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()
        self._figure_counter = 0
    
    def _setup_style(self):
        """Matplotlib stilini ayarlar"""
        sns.set_style(self.config.figure_style)
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['figure.dpi'] = self.config.figure_dpi
        plt.rcParams['font.size'] = 10
        self.colors = sns.color_palette(self.config.color_palette, 10)
    
    def _save_and_show(self, fig, filename: str):
        """Figürü kaydeder ve/veya gösterir"""
        if self.config.save_figures:
            self._figure_counter += 1
            save_path = self.output_dir / f"{self._figure_counter:02d}_{filename}.png"
            fig.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            print(f"   [PLOT] Saved: {save_path.name}")
        
        if self.config.show_figures:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_boxplots(self, machines: List[Dict], title: str = "Machine Times Boxplot"):
        """Tüm makineler için kutu grafiği çizer"""
        fig, ax = plt.subplots(figsize=(max(12, len(machines) * 1.5), 6))
        
        data = [m['data'] for m in machines]
        labels = [m['name'] for m in machines]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        for i, (box, color) in enumerate(zip(bp['boxes'], self.colors)):
            box.set(facecolor=color, alpha=0.6)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Machine / Process', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        self._save_and_show(fig, "boxplots")
    
    def plot_distribution_fit(self, data: np.ndarray, results: List[Dict], 
                               machine_name: str) -> None:
        """Dağılım uydurma grafiğini çizer (V2: İstatistiksel bilgi ağırlıklı)"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Sol: Histogram ve uydurulmuş dağılımlar
        ax1 = axes[0]
        sns.histplot(data, stat="density", kde=False, color="skyblue", 
                     alpha=0.6, label="Observed Data", ax=ax1)
        
        x = np.linspace(min(data), max(data), 200)
        
        # En iyi 3 dağılımı çiz
        for i, result in enumerate(results[:3]):
            dist = getattr(stats, result['distribution'])
            pdf = dist.pdf(x, *result['params'])
            h0 = result['hypothesis_decision']
            h0_marker = "✓" if not h0['rejected'] else "✗"
            ax1.plot(x, pdf, lw=2, color=self.colors[i],
                    label=f"{result['display_name']} (p={result['ks_p_value']:.3f}) {h0_marker}")
        
        # V2: Başlıkta istatistiksel bilgi
        best = results[0] if results else None
        if best:
            h0 = best['hypothesis_decision']
            title = (
                f"{machine_name}\n"
                f"Best: {best['display_name']} | "
                f"H₀: {h0['label']} (p={h0['p_value']:.4f}, α={h0['alpha']:.2f})"
            )
        else:
            title = machine_name
        
        ax1.set_title(title, fontsize=11, fontweight='bold')
        ax1.set_xlabel("Time (seconds)", fontsize=10)
        ax1.set_ylabel("Probability Density", fontsize=10)
        ax1.legend(loc='upper right', fontsize=8)
        
        # Sağ: Q-Q Plot
        ax2 = axes[1]
        if best:
            dist = getattr(stats, best['distribution'])
            stats.probplot(data, dist=dist, sparams=best['params'], plot=ax2)
            ax2.set_title(
                f"Q-Q Plot: {best['display_name']}\n"
                f"AICc={best['aicc']:.2f}, ΔAICc={best['delta_aicc']:.2f}",
                fontsize=11
            )
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        safe_name = "".join(c if c.isalnum() else "_" for c in machine_name)
        self._save_and_show(fig, f"dist_fit_{safe_name}")
    
    def plot_summary_comparison(self, all_results: Dict) -> None:
        """Tüm makinelerin karşılaştırmalı özet grafiğini çizer"""
        machines = list(all_results.keys())
        means = [all_results[m]['descriptive']['mean'] for m in machines]
        stds = [all_results[m]['descriptive']['std'] for m in machines]
        ci_lowers = [all_results[m]['descriptive']['ci_lower'] for m in machines]
        ci_uppers = [all_results[m]['descriptive']['ci_upper'] for m in machines]
        
        fig, ax = plt.subplots(figsize=(max(12, len(machines) * 1.5), 6))
        
        x = np.arange(len(machines))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=self.colors[:len(machines)], 
                      alpha=0.7, edgecolor='black')
        
        for i, (ci_l, ci_u) in enumerate(zip(ci_lowers, ci_uppers)):
            ax.plot([i, i], [ci_l, ci_u], 'k-', lw=2)
            ax.plot([i-0.1, i+0.1], [ci_l, ci_l], 'k-', lw=2)
            ax.plot([i-0.1, i+0.1], [ci_u, ci_u], 'k-', lw=2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(machines, rotation=45, ha='right')
        ax.set_xlabel('Machine / Process', fontsize=12)
        ax.set_ylabel('Mean Time (seconds)', fontsize=12)
        ax.set_title('Machine Times Comparison\n(Bars: ±1 Std, Lines: 95% Confidence Interval)', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self._save_and_show(fig, "summary_comparison")

    def plot_model_comparison_chart(self, all_results: Dict) -> None:
        """Tüm makineler için AICc-tabanlı model karşılaştırma grafiği"""
        # Her makine için en iyi dağılımın ΔAICc farkını göster
        machines_with_fits = {m: r for m, r in all_results.items() 
                             if r.get('distribution_fits')}
        
        if not machines_with_fits:
            return
        
        n_machines = len(machines_with_fits)
        fig, axes = plt.subplots(1, min(n_machines, 4), 
                                  figsize=(5 * min(n_machines, 4), 5),
                                  squeeze=False)
        
        for idx, (name, result) in enumerate(machines_with_fits.items()):
            if idx >= 4:  # max 4 panel
                break
            ax = axes[0][idx]
            fits = result['distribution_fits']
            
            dist_names = [f['display_name'] for f in fits]
            delta_values = [f['delta_aicc'] for f in fits]
            colors = ['#2ecc71' if d <= 2 else '#f39c12' if d <= 7 else '#e74c3c' 
                      for d in delta_values]
            
            bars = ax.barh(range(len(dist_names)), delta_values, color=colors, alpha=0.8)
            ax.set_yticks(range(len(dist_names)))
            ax.set_yticklabels(dist_names, fontsize=9)
            ax.set_xlabel('ΔAICc', fontsize=10)
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5, label='ΔAICc=2')
            ax.invert_yaxis()
            
            # Değerleri barların üzerine yaz
            for bar, val in zip(bars, delta_values):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}', va='center', fontsize=8)
        
        plt.suptitle('Model Comparison: ΔAICc (lower = better fit)\n'
                     'Green: Strong (≤2) | Orange: Moderate/Weak (2-7) | Red: Very Weak (>7)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_and_show(fig, "model_comparison_aicc")


# =============================================================================
# RAPOR OLUŞTURMA SINIFI (V2)
# =============================================================================

class ReportGenerator:
    """Analiz raporu oluşturma sınıfı - V2: İstatistiksel yorumlama ağırlıklı"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) if config.output_dir else Path.cwd()
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_descriptive_table(self, all_results: Dict) -> pd.DataFrame:
        """Betimsel istatistik tablosu oluşturur"""
        rows = []
        for machine_name, result in all_results.items():
            desc = result['descriptive']
            rows.append({
                'Machine': machine_name,
                'Observations': int(desc['count']),
                'Mean': round(desc['mean'], 3),
                'Std Dev': round(desc['std'], 3),
                'CV (%)': round(desc['cv_percent'], 2),
                'Min': round(desc['min'], 3),
                'Max': round(desc['max'], 3),
                'Median': round(desc['median'], 3),
                'CI Lower': round(desc['ci_lower'], 3),
                'CI Upper': round(desc['ci_upper'], 3)
            })
        
        return pd.DataFrame(rows).set_index('Machine')
    
    def generate_distribution_table(self, all_results: Dict) -> pd.DataFrame:
        """V2: Dağılım analizi sonuç tablosu (hipotez testi vurgulu)"""
        rows = []
        for machine_name, result in all_results.items():
            best = result.get('best_distribution')
            fits = result.get('distribution_fits', [])
            tied = [f for f in fits if f.get('delta_aicc', 99) <= 2]
            
            if best:
                h0 = best['hypothesis_decision']
                rows.append({
                    'Machine': machine_name,
                    'Best Distribution': best['display_name'],
                    'KS D-Stat': round(best['ks_statistic'], 4),
                    'KS p-value': round(best['ks_p_value'], 4),
                    'H₀ Decision': h0['label'],
                    'AICc': round(best['aicc'], 2),
                    'Tied Models': len(tied),
                    'Evidence': h0['confidence_note_en'],
                })
        
        return pd.DataFrame(rows).set_index('Machine')
    
    def generate_outlier_table(self, all_results: Dict) -> pd.DataFrame:
        """Aykırı değer analizi tablosu oluşturur"""
        rows = []
        for machine_name, result in all_results.items():
            outlier = result.get('outliers', {})
            rows.append({
                'Machine': machine_name,
                'Method': outlier.get('method', '-'),
                'Outlier Count': outlier.get('outlier_count', 0),
                'Outlier (%)': round(outlier.get('outlier_percent', 0), 2),
                'Lower Bound': round(outlier.get('lower_bound', 0), 3) if outlier.get('lower_bound') else '-',
                'Upper Bound': round(outlier.get('upper_bound', 0), 3) if outlier.get('upper_bound') else '-'
            })
        
        return pd.DataFrame(rows).set_index('Machine')
    
    def generate_comprehensive_summary_table(self, all_results: Dict) -> pd.DataFrame:
        """V2: Kapsamlı özet tablosu (hipotez testi & AICc vurgulu)"""
        rows = []
        for machine_name, result in all_results.items():
            desc = result['descriptive']
            best_dist = result.get('best_distribution', {})
            outliers = result.get('outliers', {})
            fits = result.get('distribution_fits', [])
            
            h0_label = "-"
            if best_dist and 'hypothesis_decision' in best_dist:
                h0_label = best_dist['hypothesis_decision']['label']
            
            tied = len([f for f in fits if f.get('delta_aicc', 99) <= 2])

            rows.append({
                'Machine / Process': machine_name,
                'N': int(desc['count']),
                'Mean': round(desc['mean'], 3),
                'Std Dev': round(desc['std'], 3),
                'Min': round(desc['min'], 3),
                'Max': round(desc['max'], 3),
                'Best Dist.': best_dist.get('display_name', '-') if best_dist else '-',
                'H₀ Decision': h0_label,
                'KS p-value': round(best_dist.get('ks_p_value', 0), 4) if best_dist else '-',
                'AICc': round(best_dist.get('aicc', 0), 2) if best_dist else '-',
                'Tied': tied,
                'Outliers': outliers.get('outlier_count', 0)
            })
            
        return pd.DataFrame(rows).set_index('Machine / Process')
    
    def export_results_to_excel(self, all_results: Dict, filename: str = "analysis_results.xlsx"):
        """Tüm sonuçları Excel dosyasına kaydeder"""
        output_path = self.output_dir / filename
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            comp_table = self.generate_comprehensive_summary_table(all_results)
            comp_table.to_excel(writer, sheet_name='Comprehensive Summary')

            desc_table = self.generate_descriptive_table(all_results)
            desc_table.to_excel(writer, sheet_name='Descriptive Statistics')
            
            dist_table = self.generate_distribution_table(all_results)
            if not dist_table.empty:
                dist_table.to_excel(writer, sheet_name='Distribution Analysis')
            
            outlier_table = self.generate_outlier_table(all_results)
            outlier_table.to_excel(writer, sheet_name='Outlier Analysis')
            
            # V2: Yorum cümleleri sayfası
            interp_rows = []
            for machine_name, result in all_results.items():
                interp = result.get('interpretation', {})
                if interp:
                    interp_rows.append({
                        'Machine': machine_name,
                        'Interpretation (TR)': interp.get('tr', '-'),
                        'Interpretation (EN)': interp.get('en', '-'),
                    })
            if interp_rows:
                interp_df = pd.DataFrame(interp_rows).set_index('Machine')
                interp_df.to_excel(writer, sheet_name='Interpretations')
        
        print(f"\n[EXPORT] Results saved: {output_path}")
        return output_path
    
    def print_machine_analysis(self, machine_name: str, result: Dict, 
                                fitter: 'DistributionFitter') -> None:
        """V2: Tek bir makine için 3 katmanlı istatistiksel analiz çıktısı"""
        alpha = self.config.significance_level
        
        print(f"\n{'='*75}")
        print(f"  DISTRIBUTION ANALYSIS: {machine_name}")
        print(f"{'='*75}")
        
        fits = result.get('distribution_fits', [])
        desc = result['descriptive']
        
        # ══════════════════════════════════════════════════════════════════
        # KATMAN 1: Hipotez Testi Sonuçları
        # ══════════════════════════════════════════════════════════════════
        print(f"\n  ┌─ LAYER 1: Hypothesis Test Results (KS Test, α={alpha})")
        print(f"  │")
        print(f"  │  H₀: Data follows the specified distribution")
        print(f"  │  H₁: Data does NOT follow the specified distribution")
        print(f"  │")
        print(f"  │  {'Distribution':<15} {'D-Stat':<10} {'p-value':<10} {'Decision':<16} {'Evidence'}")
        print(f"  │  {'─'*75}")
        
        for fit in fits:
            h0 = fit['hypothesis_decision']
            marker = "✓" if not h0['rejected'] else "✗"
            print(f"  │  {marker} {fit['display_name']:<13} "
                  f"{fit['ks_statistic']:<10.4f} "
                  f"{fit['ks_p_value']:<10.4f} "
                  f"{h0['label']:<16} "
                  f"{h0['confidence_note_en']}")
        
        print(f"  │")
        print(f"  └─────────────────────────────────────────────────────────")
        
        # ══════════════════════════════════════════════════════════════════
        # KATMAN 2: AICc Model Karşılaştırma
        # ══════════════════════════════════════════════════════════════════
        print(f"\n  ┌─ LAYER 2: Model Comparison (AICc - lower is better)")
        print(f"  │")
        print(f"  │  Burnham & Anderson (2002) criteria:")
        print(f"  │  ΔAICc ≤ 2: Strong support | 2-4: Moderate | 4-7: Weak | >7: Very Weak")
        print(f"  │")
        print(f"  │  {'Distribution':<15} {'AICc':<12} {'ΔAICc':<10} {'BIC':<12} {'Support'}")
        print(f"  │  {'─'*65}")
        
        for fit in fits:
            marker = "★" if fit['delta_aicc'] == 0 else " "
            print(f"  │  {marker} {fit['display_name']:<13} "
                  f"{fit['aicc']:<12.2f} "
                  f"{fit['delta_aicc']:<10.2f} "
                  f"{fit['bic']:<12.2f} "
                  f"{fit['model_support']}")
        
        # Eşdeğer modelleri belirt
        tied = fitter.get_tied_candidates(fits)
        if len(tied) > 1:
            tied_names = ", ".join([f['display_name'] for f in tied])
            print(f"  │")
            print(f"  │  ⚠ {len(tied)} models are statistically tied (ΔAICc ≤ 2): {tied_names}")
        
        print(f"  │")
        print(f"  └─────────────────────────────────────────────────────────")
        
        # ══════════════════════════════════════════════════════════════════
        # KATMAN 3: Doğal Dilde Yorum
        # ══════════════════════════════════════════════════════════════════
        best = result.get('best_distribution')
        if best:
            interp = fitter.generate_interpretation(
                machine_name, best, desc, len(tied)
            )
            # Yorum cümlesini result'a kaydet (Excel export için)
            result['interpretation'] = interp
            
            print(f"\n  ┌─ LAYER 3: Statistical Interpretation")
            print(f"  │")
            # Türkçe yorum
            # Satırı 70 karakterde kes
            tr_text = interp['tr']
            words = tr_text.split()
            line = "  │  "
            for word in words:
                if len(line) + len(word) + 1 > 80:
                    print(line)
                    line = "  │  " + word
                else:
                    line = line + " " + word if line.strip().startswith("│") and len(line) > 5 else line + word
            if line.strip():
                print(line)
            
            print(f"  │")
            print(f"  └─────────────────────────────────────────────────────────")
        
        # Parametreler
        if best:
            print(f"\n  Selected: {best['display_name'].upper()}")
            print(f"  Parameters:")
            for pname, pval in zip(best['param_names'], best['params']):
                print(f"    {pname} = {pval:.6f}")
            print(f"  Data: Mean={desc['mean']:.3f}, Std={desc['std']:.3f}, N={desc['count']}")


# =============================================================================
# ANA ANALİZ SINIFI (V2)
# =============================================================================

class DataAnalysisPipeline:
    """Ana analiz pipeline sınıfı - V2"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.loader = DataLoader(self.config)
        self.analyzer = StatisticalAnalyzer(self.config)
        self.fitter = DistributionFitter(self.config)
        self.visualizer = Visualizer(self.config)
        self.reporter = ReportGenerator(self.config)
        self.results = {}
    
    def run(self, filepath: str) -> Dict:
        """Tam analiz pipeline'ını çalıştırır"""
        start_time = datetime.now()
        
        # 1. Veri Yükleme
        print("\n" + "#"*60)
        print("LOADING DATA")
        print("#"*60)
        
        self.loader.load_from_file(filepath)
        
        if not self.loader.machines:
            print("[!] No suitable data found for analysis.")
            return {}
        
        # 2. Betimsel İstatistik Analizi
        print("\n" + "#"*60)
        print("DESCRIPTIVE STATISTICS ANALYSIS")
        print("#"*60)
        
        for machine in self.loader.machines:
            name = machine['name']
            data = machine['data']
            
            self.results[name] = {
                'descriptive': self.analyzer.calculate_descriptive_stats(data),
                'normality': self.analyzer.test_normality(data),
                'outliers': self.analyzer.detect_outliers(data),
                'data': data
            }
        
        desc_table = self.reporter.generate_descriptive_table(self.results)
        print("\n")
        display_cols = ['Observations', 'Mean', 'Std Dev', 'CV (%)', 'Min', 'Max', 'CI Lower', 'CI Upper']
        print(desc_table[display_cols].to_string())
        
        # 3. Kutu Grafiği
        self.visualizer.plot_boxplots(self.loader.machines, "Machine Times Boxplot")
        
        # 4. Aykırı Değer Tablosu
        print("\n" + "#"*60)
        print("OUTLIER ANALYSIS")
        print("#"*60)
        outlier_table = self.reporter.generate_outlier_table(self.results)
        print("\n")
        print(outlier_table.to_string())
        
        # 5. Dağılım Uydurma Analizi (V2: 3 Katmanlı)
        print("\n" + "#"*60)
        print("DISTRIBUTION FITTING ANALYSIS (V2 - Statistical Reporting)")
        print("#"*60)
        
        print("\nMethodology:")
        print("  • Kolmogorov-Smirnov (KS) Test: Goodness-of-fit hypothesis test")
        print("  • Anderson-Darling (AD) Test: Enhanced tail sensitivity")
        print("  • AICc (corrected AIC): Small-sample model comparison")
        print("  • ΔAICc: Burnham & Anderson (2002) model support criteria")
        
        for machine in self.loader.machines:
            name = machine['name']
            data = machine['data']
            
            if len(data) < self.config.min_observations:
                print(f"\n[!] {name}: Insufficient observations ({len(data)} < {self.config.min_observations})")
                continue
            
            # Dağılım uydurma
            fit_results = self.fitter.fit_distributions(data)
            best_fit = self.fitter.get_best_fit(fit_results)
            
            self.results[name]['distribution_fits'] = fit_results
            self.results[name]['best_distribution'] = best_fit
            
            # V2: 3 katmanlı detaylı analiz çıktısı
            self.reporter.print_machine_analysis(name, self.results[name], self.fitter)
            
            # Görselleştirme (V2: istatistiksel bilgi ağırlıklı başlıklar)
            self.visualizer.plot_distribution_fit(data, fit_results, name)
        
        # 6. Nihai Özet
        print("\n" + "#"*60)
        print("FINAL SUMMARY TABLE (V2 - Hypothesis Test Decisions)")
        print("#"*60)
        
        dist_table = self.reporter.generate_distribution_table(self.results)
        print("\n")
        print(dist_table.to_string())
        
        # 7. Karşılaştırmalı Grafikler
        self.visualizer.plot_summary_comparison(self.results)
        
        # V2: AICc model karşılaştırma grafiği
        self.visualizer.plot_model_comparison_chart(self.results)
        
        # 8. Sonuçları Excel'e kaydet
        if self.config.export_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.reporter.export_results_to_excel(self.results, f"analysis_results_v2_{timestamp}.xlsx")
        
        # Süre bilgisi
        elapsed = datetime.now() - start_time
        print(f"\n[OK] Analysis completed! (Time: {elapsed.total_seconds():.2f} seconds)")
        
        return self.results


# =============================================================================
# KULLANIM
# =============================================================================

def main():
    """Ana fonksiyon"""
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          DATA ANALYSIS PIPELINE - V2                          ║
║          Statistical Reporting Edition                        ║
╠═══════════════════════════════════════════════════════════════╣
║  V2 Features:                                                 ║
║  • Hypothesis test decisions (H₀ rejected / not rejected)     ║
║  • AICc model comparison with ΔAICc support levels            ║
║  • Natural language interpretation for each machine           ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir_path = os.path.join(script_dir, "IDA_Bitirme_V2")

    default_filename = "Bitirme Data_V1.xlsx"
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        if os.path.exists(default_filename):
            filepath = default_filename
            print(f"[*] File found (CWD): {filepath}")
        else:
            filepath = os.path.join(script_dir, default_filename)
            
            if os.path.exists(filepath):
                print(f"[*] File found (Script Dir): {filepath}")
            else:
                print("\nPlease enter the path of the file to be analyzed:")
                print("   (Example: C:\\Users\\...\\data.xlsx)")
                filepath = input("\n   File Path: ").strip()
        
        if not filepath:
            print("\n[!] No file path specified. Exiting...")
            return None
    
    # Konfigürasyon
    # Not: Sadece belirli bir sayfayı okumak isterseniz 'sheet_name' parametresini kullanabilirsiniz:
    # sheet_name='Sheet1' veya sheet_name='Üretim_V1' gibi. (Varsayılan: 0 = İlk sayfa)
    config = AnalysisConfig(
        min_observations=20,
        significance_level=0.05,
        outlier_method='iqr',
        iqr_multiplier=1.5,
        confidence_level=0.95,
        save_figures=True,
        show_figures=False,
        export_results=True,
        output_dir=output_dir_path,
        sheet_name='Üretim_V1'  # BURAYI DEĞİŞTİREBİLİRSİNİZ (Örn: 'Üretim_V1')
    )
    
    pipeline = DataAnalysisPipeline(config)
    
    try:
        results = pipeline.run(filepath)
        return results
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {filepath}")
        return None
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        raise


if __name__ == "__main__":
    results = main()
