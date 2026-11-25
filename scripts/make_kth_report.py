"""Build a three-page research sample from existing measurements; no new experiment.

Run from the project root. Requires reportlab (document-only dependency), plus
the project's NumPy/SciPy/pandas/Matplotlib. Raw facial images are never included.
"""
from pathlib import Path
import hashlib
import json
import re
import xml.sax.saxutils as xml

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    KeepTogether,
)
from src.signals import spectrum

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/kth_research_sample'
OUT.mkdir(parents=True, exist_ok=True)
navy = '#17384A'
teal = '#137C87'
grey = '#51616A'
styles = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=21,
                            leading=24, textColor=navy, spaceAfter=5),
    'subtitle': ParagraphStyle('subtitle', fontSize=11, leading=14,
                               textColor=teal, spaceAfter=9),
    'meta': ParagraphStyle('meta', fontSize=8.4, leading=11, textColor=grey,
                           spaceAfter=8),
    'heading': ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=11,
                              leading=14, textColor=navy, spaceBefore=10, spaceAfter=5),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=9.5,
                           leading=12.6, spaceAfter=6, alignment=TA_LEFT),
    'caption': ParagraphStyle('caption', fontSize=8, leading=10.5,
                              textColor=grey, spaceAfter=6),
    'cell': ParagraphStyle('cell', fontSize=8.3, leading=10.5),
    'cellhead': ParagraphStyle('cellhead', fontName='Helvetica-Bold', fontSize=8.3,
                               leading=10.5, textColor=colors.white),
    'reference': ParagraphStyle('reference', fontSize=7.8, leading=10,
                                spaceAfter=3),
}
story, markdown = [], []


def p(text, kind='body'):
    story.append(Paragraph(text, styles[kind]))
    prefix = '# ' if kind == 'title' else ('## ' if kind == 'heading' else '')
    markdown.append(prefix + re.sub('<[^>]+>', '', text.replace('<br/>', ' ')))


def section(title):
    p(title, 'heading')


def table(headers, rows, widths):
    cells = [[Paragraph(xml.escape(str(v)), styles['cellhead']) for v in headers]]
    cells += [[Paragraph(xml.escape(str(v)), styles['cell']) for v in row] for row in rows]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(navy)),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F1F5F6'), colors.white]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 7), ('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#BAC8CE')),
    ]))
    story.append(t)
    story.append(Spacer(1,5))
    markdown.append('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---']*len(headers)) + ' |\n' +
                    '\n'.join('| ' + ' | '.join(map(str,row)) + ' |' for row in rows))


private = pd.read_csv(ROOT/'results/private/garmin_15_120/results.csv')
expanded = pd.read_csv(ROOT/'results/private/expanded_validation/per_subject_summary.csv')
identity = pd.read_csv(ROOT/'results/private/identity_cohort/summary.csv')
bp = np.load(ROOT/'results/private/garmin_15_120/bvp.npz')
fps = float(bp['fps'])
a, n = int((32-17)*fps), int(30*fps)
plt.rcParams.update({'font.size':8, 'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.labelcolor':navy, 'text.color':navy})
fig, axes = plt.subplots(1,2,figsize=(7.05,1.85),layout='constrained',
                         gridspec_kw={'width_ratios':[1,1.15]})
peak_checks = {}
for kind, label, color in [('original','Original',navy),('blur','Blur',teal)]:
    sig = bp['POS_'+kind][a:a+n]
    t = np.arange(len(sig))/fps+32
    axes[0].plot(t, sig, label=label, color=color, lw=.9, alpha=.85)
    f, power = spectrum(sig,fps)
    axes[1].plot(f*60,power,color=color,lw=1.2,label=label)
    valid = (f>=.7)&(f<=3)
    peak_checks[kind] = float(f[valid][np.argmax(power[valid])]*60)
axes[0].set(xlim=(32,38),xlabel='Source-video time (s)',ylabel='Standardized rPPG',title='A  Waveform excerpt')
axes[1].set(xlim=(42,110),xlabel='Frequency expressed as BPM',ylabel='Welch PSD',title='B  Full 30-s window spectrum')
axes[1].legend(frameon=False,fontsize=7,loc='upper right')
for hz, text, y in [(44.82421875,'44.8',1.57),(64.599609375,'64.6',1.91)]:
    axes[1].text(hz+1,y,text,fontsize=7)
for ax in axes: ax.grid(alpha=.15)
figure = OUT/'peak_switch.png'
fig.savefig(figure,dpi=230)
plt.close(fig)
assert abs(peak_checks['original']-64.599609375)<1e-8
assert abs(peak_checks['blur']-44.82421875)<1e-8

p('Facial-Video rPPG Under<br/>Privacy Transformations', 'title')
p('Signal preservation, quality warnings and reference uncertainty', 'subtitle')
p('<b>Hadi Rezaeikarjani</b> | MSc student, Biomedical Engineering, University of Oulu<br/>'
  'Independent project technical report | 13 November 2026 | KTH application PA-2026-3731', 'meta')
p('<b>Abstract.</b> This exploratory study tests whether facial blur and pixelation preserve '
  'camera-derived pulse information. A CPU pipeline combines face-region tracking, classical rPPG '
  'extraction and spectral heart-rate estimation. Evaluation uses one consented recording and '
  'three public UBFC-rPPG participants. Stable whole-record heart rate conceals a local peak switch '
  'after blur; further recordings expose detector sensitivity, imperfect quality warnings and '
  'uncertainty in the contact-PPG reference itself. The contribution is a reproducible baseline '
  'and failure analysis, with explicit separation of signal consistency, reference agreement and '
  'physiological accuracy. No clinical or biometric-anonymity claim is made.')

section('1  Research question and experimental design')
p('Can simple facial de-identification reduce identity-related similarity while preserving '
  'remote photoplethysmography (rPPG) and estimated heart rate (HR)? This complements learned '
  'privacy-preserving video research [1] with transparent classical baselines. For cardiovascular '
  'monitoring, a second question is whether a plausible HR estimate remains trustworthy when '
  'the underlying waveform or measurement conditions change.')
p('The consented recording was cropped to source-video seconds 15-120 and resampled to 30 fps '
  'for processing. Its Garmin Epix Gen 2 export contains only 13 processed HR records, with gaps '
  'up to 29 s, rather than raw PPG. Audible start/end tones support approximate alignment, but '
  'unknown device-processing latency remains. UBFC-rPPG Dataset 2 [2] provides facial video, '
  'contact PPG and device HR for subjects 1, 3 and 4. Subject 1 was a development case; subjects '
  '3/4 were selected before their outcomes, with signal and warning settings held fixed. '
  'They provide two overlapping 30-s windows and one 30-s window respectively: three temporal '
  'intervals from two new participants, not independent observations for every method/variant.')

section('2  Offline processing and paired preservation measurements')
p('OpenCV Haar detection with Lucas-Kanade tracking, or a separate per-frame YuNet detector, '
  'defines approximate forehead and cheek patches. Equal-weight patch RGB means enter '
  'Plane-Orthogonal-to-Skin (POS) [3] or a simplified CHROM implementation [4], using 1.6-s '
  'temporal normalization and overlap-add. Signals are detrended, filtered with a third-order '
  'zero-phase Butterworth bandpass (0.7-3 Hz), standardized and trimmed by 2 s at each edge. '
  'HR is 60 times the dominant Welch frequency; 10-s segments and a 4096-point FFT are used. '
  'Zero padding does not create additional physiological resolution. Analysis windows are '
  '30 s with a 15-s step. Zero-phase filtering makes this an offline pipeline.')
p('Face crops receive Gaussian blur (sigma = 0.10 times the smaller face dimension), 8 x 8 '
  'pixelation, or black masking as a negative control. Original-frame geometry is reused across '
  'variants, isolating image transformation from detection failure. HR differences and signed '
  'zero-lag Pearson correlation measure preservation against the original, not accuracy against '
  'independent physiology. Masked traces are invalid and never assigned zero error.')
rows = []
for m in ['POS','CHROM']:
    d = private[private.method==m].set_index('variant')
    rows.append([m, f"{d.loc['original','hr_bpm']:.2f}", f"{d.loc['blur','hr_bpm']:.2f}",
                 f"{d.loc['pixelate','hr_bpm']:.2f}",
                 f"{d.loc['blur','pearson_original']:.3f} / {d.loc['pixelate','pearson_original']:.3f}"])
table(['Method','Original HR','Blur HR','Pixel HR','Correlation: blur / pixel'],rows,[58,81,70,70,228.28])
p('<b>Table 1.</b> Whole-record HR (BPM) and waveform preservation, consented recording. '
  'Near-equal HR values do not establish local waveform fidelity or clinical accuracy.', 'caption')

story.append(PageBreak())
section('3  A local failure hidden by whole-record agreement')
story.append(Image(str(figure),width=507.28,height=133.11))
markdown.append('![Figure 1: waveform and spectral peak switch](peak_switch.png)')
p('<b>Figure 1.</b> POS, source seconds 32-62. Only six seconds of waveform are displayed; '
  'the spectrum uses the full 30-s window (display restricted to 42-110 BPM). Blur changes '
  'the strongest component from 64.60 to 44.82 BPM. Both components already exist in the '
  'original spectrum; the algorithm selects the strongest rather than moving a peak.', 'caption')
p('Whole-record POS HR changes by only 0.44 BPM after blur (Table 1), yet this local window '
  'changes by 19.78 BPM. At approximately 64.6 and 44.8 BPM, original spectral powers are '
  '1.915 and 1.145; after blur they are 1.380 and 1.427. This demonstrates a change in '
  'relative component strength, not proof of which component is physiological. Sparse watch '
  'HR provides contextual agreement only; it cannot validate beat timing or waveform morphology.')

section('4  New-recording results and exploratory quality warnings')
p('The public-data target is the dominant frequency of timestamp-resampled contact PPG, '
  'processed with the same band and Welch settings. MAE = mean absolute difference between '
  'window HR and this fixed spectral-reference estimate. Table 2 retains both detector '
  'conditions; no best detector is selected per person. All new-recording runs completed '
  'without manual regions. On development subject 1, strict Haar/LK failed; its manual-region '
  'fallback is a separate condition and is not mixed into this table.')
rows=[]
for subject in ['subject3','subject4']:
    for detector in ['haar','yunet']:
        for m in ['POS','CHROM']:
            d=expanded[(expanded.subject==subject)&(expanded.detector==detector)&(expanded.method==m)].set_index('variant')
            rows.append([subject.replace('subject','S'), 'Haar/LK' if detector=='haar' else 'YuNet',m,
                         str(int(d.loc['original','total_windows'])),
                         *[f"{d.loc[v,'hr_mae_bpm']:.2f}" for v in ['original','blur','pixelate']]])
table(['Person','Detector','Method','Windows','Original','Blur','Pixel'],rows,[49,79,59,65,88,83,84.28])
p('<b>Table 2.</b> MAE in BPM versus a fixed contact-PPG spectral-reference estimate. '
  'Conditions share windows and reference signals. A 0.00 value can reflect a shared '
  'spectral bin; it does not establish perfect physiological accuracy.', 'caption')
p('An exploratory warning marks strongest-to-distinct-competing-peak power ratio &lt;1.2 '
  '(minimum separation 0.15 Hz), or a dominant peak within 0.05 Hz of a band edge. Invalid '
  'signals are flagged separately. These heuristics were chosen after the consented-recording '
  'failure, then frozen before subjects 3/4; they neither correct HR nor estimate a probability '
  'of accuracy. The blur failure is flagged (ratio 1.035), but CHROM in the same interval is '
  'also flagged despite approximate watch agreement.')
p('For deviations &gt;5 BPM from the fixed reference, Haar/LK produces TP/FP/FN/TN = '
  '4/5/3/6, and YuNet 2/6/1/9. Each set contains 18 correlated assessments across only '
  'three intervals; six assessments share one ambiguous reference interval. Warnings '
  'therefore miss some large deviations and flag some near-reference estimates. These '
  'counts are descriptive, not validated classifier performance. Baseline estimates and '
  'all unfavourable windows remain in the analysis.')

story.append(PageBreak())
section('5  Reference uncertainty changes the interpretation of error')
p('Subject 3 has competing contact-PPG components. A post-hoc audit compared the original '
  'Welch estimate, native-time Lomb-Scargle, detected pulse intervals and device HR. Twelve '
  'filter/polarity/prominence combinations are retained; none was chosen to minimize rPPG error.')
table(['Contact/device summary (BPM)','Window 1 (~2-32 s)','Window 2 (~17-47 s)'],[
    ['Fixed Welch dominant peak','87.43','84.84'],
    ['Median detected interval rate','98.49','98.49'],
    ['Pulse count per detected span','94.28-94.61','95.89-96.17'],
    ['Timestamp-weighted device HR','100.95','104.56'],
],[242,132.64,132.64])
p('<b>Table 3.</b> Reference sensitivity on subject 3. These statistics have different '
  'definitions; none is independently adjudicated physiological truth.', 'caption')
p('Native-time Lomb-Scargle selects 101.40/89.82 BPM. Differences in filtering, segmentation '
  'and sampling prevent isolating interpolation as a cause. Video/reference endpoints differ '
  'by less than 0.15 s, without an obvious gross duration-scale mismatch; other timing '
  'uncertainty remains. No reference was replaced or lag fitted. Table 2 and warning counts '
  'remain conditional on the original reference. This uncertainty does not explain away '
  'Haar POS estimates near 48 BPM.')

section('6  Privacy findings and limits')
p('A separate CPU SFace experiment uses three UBFC identities, enrolling at 5/10/15 s and '
  'probing at 30/35/40 s within the same session. Mean genuine cosine falls from 0.984 '
  '(original) to 0.125 (blur) and 0.202 (pixelation), while own-template rank-1 matches '
  'remain 9/9, 6/9 and 8/9. Original landmarks assist alignment; transformed-face detection '
  'fails on all nine probes per variant. Masking still yields 6/9 forced closed-set matches, '
  'possibly reflecting residual context or gallery bias. These small-gallery results '
  'prove neither anonymity nor general recognition performance, and do not constitute a '
  'matched privacy/utility frontier with the YuNet HR runs.')

section('7  Implications for wearable cardiovascular biofeedback')
p('The transferable contribution is biosignal processing and controlled failure analysis: '
  'plausible average HR alone is insufficient to justify feedback. '
  'KTH PA-2026-3731 concerns cardiac locomotor coupling, embedded systems and human-study '
  'validation [5]. A next experiment would predefine beat/phase and HR targets, '
  'synchronize ECG or validated contact PPG with motion sensing, validate causal processing '
  'and timing latency, and evaluate withholding feedback under uncertainty. These are '
  'proposed extensions: this project validates no locomotor coupling, HRV or beat-phase '
  'accuracy and implements no wearable hardware or feedback controller.')

section('8  Reproducibility, scope and research practice')
p('The CPU pipeline ran on macOS arm64, Python 3.13.5. Unmodified pyVHR-cpu 1.0.2 '
  'classical functions ran via an isolated loader, not the full video Pipeline. '
  'rPPG-Toolbox classical POS ran at revision b7500b8 with an explicit NumPy 2 alias; '
  'no neural training or full benchmark ran. All 43 project tests pass, checking software '
  'behaviour rather than human accuracy. Configurations, hashes and logs are retained. '
  'Reproduction modules: scripts.validate_external, scripts.evaluate_quality and '
  'scripts.audit_reference; detailed commands accompany the source project.')
p('One consented recording, three public participants, overlapping windows and post-hoc '
  'diagnostics support no population-level or clinical inference. No raw videos, identifiable '
  'images or embedding vectors appear here. Development, debugging and report preparation '
  'used generative-AI assistance. This is an unpublished project sample, not a peer-reviewed publication.')

section('Selected references')
for ref in [
    '[1] Savic, M. R.; Zhao, G. De-identification of facial videos while preserving remote physiological utility. BMVC, 2023. https://proceedings.bmvc2023.org/230/',
    '[2] Bobbia, S. et al. Unsupervised skin tissue segmentation for remote photoplethysmography. Pattern Recognition Letters, 2017. Dataset: https://sites.google.com/view/ybenezeth/ubfcrppg',
    '[3] Wang, W. et al. Algorithmic Principles of Remote PPG. IEEE TBME 64(7), 1479-1491, 2017. doi:10.1109/TBME.2016.2609282.',
    '[4] de Haan, G.; Jeanne, V. Robust Pulse Rate From Chrominance-Based rPPG. IEEE TBME 60(10), 2878-2886, 2013. doi:10.1109/TBME.2013.2266196.',
    '[5] KTH. Doctoral student in Wearable Biofeedback Systems for Cardiovascular Health, PA-2026-3731. https://www.kth.se/lediga-jobb/984498?l=en (accessed 13 November 2026).',
]:
    p(xml.escape(ref),'reference')


def footer(canvas, doc):
    canvas.saveState()
    w,h=A4
    canvas.setStrokeColor(colors.HexColor('#B9C9D0'))
    canvas.line(44,34,w-44,34)
    canvas.setFont('Helvetica',7.5)
    canvas.setFillColor(colors.HexColor(grey))
    canvas.drawString(44,23,'H. Rezaeikarjani | rPPG research sample | 13 November 2026')
    canvas.drawRightString(w-44,23,f'{doc.page} / 3')
    canvas.restoreState()


pdf=OUT/'Hadi_Rezaeikarjani_rPPG_Research_Sample_KTH.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=44,leftMargin=44,
                      topMargin=33,bottomMargin=45,
                      title='Facial-Video rPPG Under Privacy Transformations',
                      author='Hadi Rezaeikarjani',
                      subject='Research sample: signal preservation and reference uncertainty')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
assert len(PdfReader(pdf).pages) == 3, 'Research sample must contain exactly three pages'
(OUT/'research_sample.md').write_text('\n\n'.join(markdown)+'\n')
sources = [
    'docs/technical_report.pdf','docs/QUALITY.md','docs/REFERENCE_AUDIT.md',
    'docs/EXPANDED_VALIDATION.md','src/signals.py','src/video.py','src/quality.py',
    'results/private/garmin_15_120/results.csv','results/private/garmin_15_120/bvp.npz',
    'results/private/expanded_validation/per_subject_summary.csv',
    'results/private/identity_cohort/summary.csv',
]
manifest = {'kind':'report revision only; original experimental outputs unchanged',
            'sources':{x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in sources},
            'figure_peak_checks_bpm':peak_checks,
            'build_dependency':'reportlab 5.1.0 (isolated document-only installation)',
            'job_source':'https://www.kth.se/lediga-jobb/984498?l=en'}
(OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2))
print(pdf)
