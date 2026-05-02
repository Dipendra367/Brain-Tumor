from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from io import BytesIO
import base64
from datetime import datetime

# ── Colors ─────────────────────────────────────────────────
DARK       = colors.HexColor('#0a0e1a')
ACCENT     = colors.HexColor('#00d4ff')
CARD       = colors.HexColor('#111827')
DANGER     = colors.HexColor('#ef4444')
WARNING    = colors.HexColor('#f59e0b')
SUCCESS    = colors.HexColor('#10b981')
PURPLE     = colors.HexColor('#7c3aed')
LIGHT_GRAY = colors.HexColor('#94a3b8')
WHITE      = colors.white

SEVERITY_COLORS = {
    'high'   : DANGER,
    'medium' : WARNING,
    'none'   : SUCCESS,
}

CLASS_INFO = {
    'Glioma'     : 'Malignant brain tumor requiring immediate medical attention and specialist review.',
    'Meningioma' : 'Usually benign tumor arising from meninges. Requires monitoring and follow-up.',
    'No Tumor'   : 'No tumor detected. Brain tissue appears normal on MRI analysis.',
    'Pituitary'  : 'Pituitary tumor detected. Often treatable with surgery or medication.',
}

RECOMMENDATIONS = {
    'Glioma'     : [
        'Urgent referral to neurosurgery and neuro-oncology',
        'Schedule contrast-enhanced MRI for surgical planning',
        'Multidisciplinary tumor board review recommended',
        'Discuss treatment options: surgery, radiation, chemotherapy',
        'Patient and family counseling regarding diagnosis',
    ],
    'Meningioma' : [
        'Referral to neurosurgery for evaluation',
        'Serial MRI monitoring every 6-12 months',
        'Assess for symptoms: headache, vision changes, seizures',
        'Discuss watchful waiting vs. surgical intervention',
        'Follow-up with neurology as appropriate',
    ],
    'No Tumor'   : [
        'No immediate intervention required',
        'Continue routine clinical follow-up',
        'Correlate with clinical symptoms if present',
        'Consider repeat imaging if symptoms persist',
        'Maintain healthy lifestyle and regular check-ups',
    ],
    'Pituitary'  : [
        'Referral to endocrinology for hormonal evaluation',
        'Ophthalmology review for visual field assessment',
        'Pituitary hormone panel blood tests recommended',
        'Discuss treatment: medication, surgery, or radiation',
        'Regular monitoring of tumor size and hormone levels',
    ],
}

# ── Page number canvas ─────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_footer(self, page_count):
        self.saveState()
        self.setFont('Helvetica', 8)
        self.setFillColor(LIGHT_GRAY)
        self.drawString(2*cm, 1.2*cm, 'BrainDetect — AI-Powered Brain Tumour Detection System')
        self.drawRightString(A4[0] - 2*cm, 1.2*cm,
                             f'Page {self._pageNumber} of {page_count}')
        self.restoreState()

# ── Main report generator ──────────────────────────────────
def generate_pdf_report(
    report_id: str,
    patient_name: str,
    patient_age: str,
    patient_gender: str,
    doctor_name: str,
    hospital: str,
    prediction_class: str,
    confidence: float,
    all_scores: dict,
    severity: str,
    gradcam_overlay_b64: str,
) -> bytes:
    """
    Generates a professional PDF diagnostic report.
    Returns raw PDF bytes.
    """
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Custom styles ──────────────────────────────────────
    title_style = ParagraphStyle('Title', parent=styles['Normal'],
        fontSize=22, fontName='Helvetica-Bold', textColor=ACCENT,
        spaceAfter=4, alignment=TA_LEFT)

    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
        fontSize=10, fontName='Helvetica', textColor=LIGHT_GRAY,
        spaceAfter=2, alignment=TA_LEFT)

    section_style = ParagraphStyle('Section', parent=styles['Normal'],
        fontSize=8, fontName='Helvetica-Bold', textColor=LIGHT_GRAY,
        spaceBefore=16, spaceAfter=8, letterSpacing=1)

    body_style = ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=10, fontName='Helvetica', textColor=colors.HexColor('#e2e8f0'),
        spaceAfter=6, leading=16)

    # ── HEADER ─────────────────────────────────────────────
    header_data = [[
        Paragraph('BrainDetect', ParagraphStyle('Logo', parent=styles['Normal'],
            fontSize=20, fontName='Helvetica-Bold', textColor=ACCENT)),
        Paragraph(
            f'<font color="#94a3b8">Report ID: </font>'
            f'<font color="#00d4ff"><b>{report_id}</b></font>',
            ParagraphStyle('ReportId', parent=styles['Normal'],
                fontSize=11, fontName='Helvetica', alignment=TA_RIGHT)),
    ]]
    header_table = Table(header_data, colWidths=[9*cm, 8*cm])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(header_table)

    story.append(Paragraph('AI-Powered Brain Tumour Detection', subtitle_style))
    story.append(HRFlowable(width='100%', thickness=1, color=ACCENT, spaceAfter=16))

    # ── PATIENT INFO TABLE ─────────────────────────────────
    story.append(Paragraph('PATIENT INFORMATION', section_style))

    now = datetime.now().strftime('%d %B %Y, %H:%M')
    info_data = [
        ['Patient Name', patient_name or '—',    'Date',     now],
        ['Age',          patient_age or '—',      'Gender',   patient_gender or '—'],
        ['Referring Doctor', doctor_name or '—',  'Hospital', hospital or '—'],
    ]

    info_table = Table(info_data, colWidths=[3.5*cm, 6*cm, 3*cm, 5*cm])
    info_table.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('TEXTCOLOR',   (0,0), (0,-1), LIGHT_GRAY),
        ('TEXTCOLOR',   (2,0), (2,-1), LIGHT_GRAY),
        ('TEXTCOLOR',   (1,0), (1,-1), colors.HexColor('#e2e8f0')),
        ('TEXTCOLOR',   (3,0), (3,-1), colors.HexColor('#e2e8f0')),
        ('FONTNAME',    (1,0), (1,-1), 'Helvetica-Bold'),
        ('FONTNAME',    (3,0), (3,-1), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#111827'), colors.HexColor('#0f1623')]),
        ('TOPPADDING',  (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 16))

    # ── DIAGNOSIS RESULT ───────────────────────────────────
    story.append(Paragraph('DIAGNOSIS RESULT', section_style))

    sev_color = SEVERITY_COLORS.get(severity, LIGHT_GRAY)

    diag_data = [[
        Paragraph(f'<b>{prediction_class}</b>',
            ParagraphStyle('DiagClass', parent=styles['Normal'],
                fontSize=24, fontName='Helvetica-Bold',
                textColor=sev_color)),
        Paragraph(
            f'<font color="#94a3b8">Confidence Score</font><br/>'
            f'<font color="#00d4ff" size="20"><b>{confidence:.1f}%</b></font>',
            ParagraphStyle('Conf', parent=styles['Normal'],
                fontSize=10, alignment=TA_RIGHT)),
    ]]
    diag_table = Table(diag_data, colWidths=[10*cm, 7*cm])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor('#111827')),
        ('TOPPADDING',    (0,0), (-1,-1), 16),
        ('BOTTOMPADDING', (0,0), (-1,-1), 16),
        ('LEFTPADDING',   (0,0), (-1,-1), 16),
        ('ROUNDEDCORNERS', [8]),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 8))

    # Clinical info
    info_text = CLASS_INFO.get(prediction_class, '')
    story.append(Paragraph(info_text, body_style))
    story.append(Spacer(1, 8))

    # ── PROBABILITY SCORES TABLE ───────────────────────────
    story.append(Paragraph('PROBABILITY DISTRIBUTION', section_style))

    score_rows = [['Tumour Class', 'Probability', 'Indicator']]
    for cls, score in all_scores.items():
        bar = '█' * int(score / 5) + '░' * (20 - int(score / 5))
        score_rows.append([cls, f'{score:.1f}%', bar[:20]])

    score_table = Table(score_rows, colWidths=[5*cm, 3*cm, 9.5*cm])
    score_table.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('TEXTCOLOR',   (0,0), (-1,0), LIGHT_GRAY),
        ('TEXTCOLOR',   (0,1), (0,-1), colors.HexColor('#e2e8f0')),
        ('TEXTCOLOR',   (1,1), (1,-1), ACCENT),
        ('TEXTCOLOR',   (2,1), (2,-1), ACCENT),
        ('FONTNAME',    (1,1), (1,-1), 'Helvetica-Bold'),
        ('BACKGROUND',  (0,0), (-1,0), colors.HexColor('#0f1623')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#111827'), colors.HexColor('#0f1623')]),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('ALIGN', (1,0), (1,-1), 'CENTER'),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 16))

    # ── GRAD-CAM IMAGE ─────────────────────────────────────
    if gradcam_overlay_b64:
        story.append(Paragraph('GRAD-CAM EXPLAINABILITY', section_style))
        story.append(Paragraph(
            'The heatmap below shows which regions of the MRI scan influenced the AI decision. '
            'Red/yellow areas indicate high importance; blue areas indicate lower importance.',
            body_style))
        story.append(Spacer(1, 8))

        try:
            img_bytes = base64.b64decode(gradcam_overlay_b64)
            img_buf   = BytesIO(img_bytes)
            img       = Image(img_buf, width=8*cm, height=8*cm)
            img.hAlign = 'CENTER'
            story.append(img)
        except Exception:
            story.append(Paragraph('Grad-CAM image unavailable.', body_style))

        story.append(Spacer(1, 16))

    # ── RECOMMENDATIONS ────────────────────────────────────
    story.append(Paragraph('AI-GENERATED RECOMMENDATIONS', section_style))
    story.append(Paragraph(
        '<i>These recommendations are generated by AI for informational purposes only. '
        'Always follow clinical judgment and specialist advice.</i>',
        ParagraphStyle('Disclaimer', parent=styles['Normal'],
            fontSize=8, fontName='Helvetica-Oblique',
            textColor=LIGHT_GRAY, spaceAfter=10)))

    recs = RECOMMENDATIONS.get(prediction_class, [])
    for i, rec in enumerate(recs, 1):
        story.append(Paragraph(f'{i}.  {rec}', body_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=colors.HexColor('#1e2d45'), spaceAfter=12))

    # ── FOOTER DISCLAIMER ──────────────────────────────────
    disclaimer = (
        'DISCLAIMER: This report is generated by the BrainDetect AI system and is intended '
        'as a decision-support tool only. It does not constitute a medical diagnosis. '
        'All findings must be confirmed by a qualified medical professional. '
        'BrainDetect AI — braindetect-1842e.web.app'
    )
    story.append(Paragraph(disclaimer,
        ParagraphStyle('FooterDisc', parent=styles['Normal'],
            fontSize=7, fontName='Helvetica', textColor=LIGHT_GRAY,
            alignment=TA_CENTER, leading=12)))

    # ── Build PDF ──────────────────────────────────────────
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.read()