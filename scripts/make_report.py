"""Render the consolidated three-page aggregate report; no participant images."""
import json,textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sections=json.loads(Path('configs/report_sections.json').read_text())
with PdfPages('docs/technical_report.pdf') as pdf:
    for page,(title,parts) in enumerate(sections,1):
        fig=plt.figure(figsize=(8.27,11.69));fig.patch.set_facecolor('white')
        fig.text(.08,.955,'Privacy-Preserving rPPG Baseline',fontsize=16,weight='bold',color='#12354b')
        fig.text(.08,.925,title,fontsize=12,weight='bold')
        fig.text(.08,.902,'Portfolio technical report | Updated 12 November 2026 | Preliminary research prototype',fontsize=8,color='#555555')
        y=.87
        for heading,body in parts:
            fig.text(.08,y,heading,fontsize=10,weight='bold',color='#12354b');y-=.021
            lines=textwrap.wrap(body,width=94)
            fig.text(.08,y,'\n'.join(lines),fontsize=9,linespacing=1.4,va='top');y-=len(lines)*.0151+.026
        assert y>.05,(page,y)
        fig.text(.08,.035,'No clinical accuracy or biometric anonymity guarantee. No participant images included.',fontsize=8,color='#555555')
        fig.text(.9,.035,str(page),fontsize=8);pdf.savefig(fig);plt.close(fig)
print('Consolidated three-page report created')
