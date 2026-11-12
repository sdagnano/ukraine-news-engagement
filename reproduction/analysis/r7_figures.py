from pathlib import Path
import json
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black, Color
ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['Likes', 'Love', 'Angry', 'Haha', 'Sad', 'Wow', 'Comments', 'Shares']
GRAY = Color(0.5, 0.5, 0.5)

def text(c, x, y, value, size=9.5, align='left', bold=False):
    c.setFillColor(black)
    c.setFont('Times-Bold' if bold else 'Times-Roman', size)
    method = {'left': c.drawString, 'right': c.drawRightString, 'center': c.drawCentredString}[align]
    method(x, y, str(value))

def axis(c, x0, x1, low, high, ticks, y, top):
    position = lambda value: x0 + (value - low) / (high - low) * (x1 - x0)
    c.setStrokeColor(black)
    c.setLineWidth(0.5)
    c.line(x0, y, x1, y)
    for value in ticks:
        x = position(value)
        c.line(x, y, x, y - 3)
        text(c, x, y - 14, value, 9, 'center')
    c.setStrokeColor(GRAY)
    c.setLineWidth(0.5)
    c.setDash(2, 2)
    c.line(position(0), y + 7, position(0), top)
    c.setDash()
    return position

def render_figures():
    channels = pd.read_csv(ROOT / 'results/r7_composition_channels.csv').set_index('channel').loc[CHANNELS]
    occurrence = pd.read_csv(ROOT / 'results/cell_OLS.csv').query("platform=='facebook' and design=='date_all' and target=='any' and block==14").set_index('channel').loc[CHANNELS]
    canonical = json.loads((ROOT / 'results/r7_composition_canonical.json').read_text())
    (ROOT / 'figures').mkdir(exist_ok=True)
    (ROOT / 'audit').mkdir(exist_ok=True)
    c = canvas.Canvas(str(ROOT / 'figures/action_margins.pdf'), pagesize=(454, 260), invariant=1)
    c.setTitle('Action occurrence and count: common source-day OLS design')
    c.setCreator('Reproducible scientific figure script')
    x_occ = axis(c, 67, 211, -8, 14, [-5, 0, 5, 10], 45, 232)
    x_count = axis(c, 290, 411, -110, 20, [-100, -50, 0], 45, 232)
    text(c, 67, 246, '(a) Occurrence', 10.5, bold=True)
    text(c, 290, 246, '(b) Count', 10.5, bold=True)
    text(c, 139, 12, 'Percentage points', 9.5, 'center')
    text(c, 350.5, 12, 'Actions per post', 9.5, 'center')
    rows = []
    for j, name in enumerate(CHANNELS):
        y = 220 - j * 22
        a = float(occurrence.loc[name, 'beta']) * 100
        b = float(channels.loc[name, 'OLS_count_beta'])
        a_label = f'{a:.2f}'
        b_label = f'{b:.3f}' if name == 'Shares' else f'{b:.2f}'
        text(c, 1, y - 3, name, 9.5)
        for position, value in [(x_occ, a), (x_count, b)]:
            c.setFillColor(black)
            c.circle(position(value), y, 2.4, fill=1, stroke=0)
        text(c, 249, y - 3, a_label, 9, 'right')
        text(c, 452, y - 3, b_label, 9, 'right')
        rows.append(dict(channel=name, occurrence_pp=a, count_actions_per_post=b, occurrence_label=a_label, count_label=b_label))
    c.showPage()
    c.save()
    q = canonical['relative_composition']
    components = [('Shares numerator', float(q['shares_log_component'])), ('Likes denominator decline', float(q['likes_decline_log_component']))]
    total = float(q['log_ratio_of_ratios'])
    c = canvas.Canvas(str(ROOT / 'figures/relative_composition.pdf'), pagesize=(454, 162), invariant=1)
    c.setTitle('Numerator and denominator components of the log ratio contrast')
    c.setCreator('Reproducible scientific figure script')
    left, right = (150, 342)
    position = lambda value: left + value / 0.26 * (right - left)
    text(c, 452, 146, 'Log component (share)', 9, 'right')
    for y, (name, value) in zip([119, 87], components):
        text(c, 1, y - 3, name, 9.5)
        c.setFillColor(Color(0.3, 0.3, 0.3))
        c.rect(left, y - 4.5, position(value) - left, 9, stroke=0, fill=1)
        text(c, 452, y - 3, f'{value:.4f} ({value / total * 100:.1f}%)', 9, 'right')
    c.setStrokeColor(black)
    c.setLineWidth(0.5)
    c.line(left, 59, right, 59)
    for value in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        x = position(value)
        c.line(x, 59, x, 56)
        text(c, x, 44, f'{value:.2f}', 8.5, 'center')
    text(c, (left + right) / 2, 27, 'Contribution to the log ratio of ratios', 9, 'center')
    text(c, 1, 4, f"Sum = {total:.4f}; ratio of ratios = {q['ratio_of_ratios']:.3f}", 9.5)
    c.showPage()
    c.save()
    payload = dict(style='Black/gray, standard serif, vector marks, no decorative elements', descriptive_points_no_uncertainty_bands=True, action_margins=rows, relative_composition=dict(components=components, total_log_contrast=total, ratio_of_ratios=q['ratio_of_ratios'], percentages_are_shares_of_log_contrast=True))
    (ROOT / 'audit/figure_restyle_plot_data.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload
if __name__ == '__main__':
    render_figures()
    print('Two vector figures redrawn from saved values; scientific results unchanged.')
