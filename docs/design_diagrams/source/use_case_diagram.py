"""Generate use_case_diagram.svg for the DR-CDSS (UML use-case diagram, black-and-white style).

    python use_case_diagram.py <out.svg>

Use cases follow the implemented API: backend/app/routers/auth.py, assessments.py and admin.py.
The Super Administrator manages accounts only; the system gives that role no clinical content.
"""
import sys

DY = 40                             # room under the title
W, H = 1120, 1210 + DY
BOX = (275, 45, 850, 1165 + DY)     # system boundary: x0, y0, x1, y1
EW, EH = 230, 62                    # use-case ellipse size
FONT = "Times New Roman, Times, serif"

physician = (78, 602 + DY)
admin = (1040, 602 + DY)
left_x, mid_x, right_x = 405, 563, 720

left = [("Request Account\n(Register)", 110), ("View Worklist", 185), ("Create Assessment", 260),
        ("Upload Fundus Image", 335), ("View Validation Results", 410), ("View Class\nProbabilities", 485),
        ("View Grad-CAM\nExplanation", 720), ("Submit Professional\nReview", 795), ("View Completed\nRecord", 870),
        ("Search Assessment\nHistory", 945), ("View Audit Trail", 1020), ("Download PDF Report", 1095)]
shared = [("Log In/Log Out", 562), ("Change Password", 643)]
right = [("View Account\nSummary", 175), ("Approve / Reject\nRegistrations", 270), ("Create Physician\nAccounts", 365),
         ("Suspend / Reactivate\nAccounts", 840), ("Reset Passwords", 935), ("View Account\nAudit Log", 1030)]
left = [(label, y + DY) for label, y in left]
shared = [(label, y + DY) for label, y in shared]
right = [(label, y + DY) for label, y in right]

out = []
add = out.append
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">')
add('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">'
    '<path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#fff"/>')
x0, y0, x1, y1 = BOX
add(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" stroke="#000" stroke-width="1.4" stroke-dasharray="5,4"/>')
for i, line in enumerate(["AI-BASED CLINICAL DECISION SUPPORT SYSTEM", "FOR EARLY DETECTION OF DIABETIC RETINOPATHY"]):
    add(f'<text x="{(x0 + x1) / 2}" y="{y0 + 26 + i * 20}" text-anchor="middle" font-size="16" font-weight="bold">{line}</text>')


def ellipse(cx, cy, label):
    add(f'<ellipse cx="{cx}" cy="{cy}" rx="{EW / 2}" ry="{EH / 2}" fill="#fff" stroke="#000" stroke-width="1.4"/>')
    lines = label.split("\n")
    for i, line in enumerate(lines):
        dy = (i - (len(lines) - 1) / 2) * 18
        add(f'<text x="{cx}" y="{cy + dy + 5}" text-anchor="middle" font-size="15" font-weight="bold">{line}</text>')


def link(actor, cx, cy):
    """Line from the actor's shoulder to the near end of the use case (its leftmost or
    rightmost point), so a line never passes through a neighbouring use case."""
    from_left = actor is physician
    ax, ay = actor[0] + (22 if from_left else -22), actor[1] - 8
    ex = cx - EW / 2 if from_left else cx + EW / 2
    add(f'<line x1="{ax}" y1="{ay}" x2="{ex}" y2="{cy}" stroke="#000" stroke-width="1.2" marker-end="url(#arrow)"/>')


def actor(x, y, name):
    add(f'<circle cx="{x}" cy="{y - 58}" r="13" fill="#fff" stroke="#000" stroke-width="1.4"/>')
    add(f'<line x1="{x}" y1="{y - 45}" x2="{x}" y2="{y + 5}" stroke="#000" stroke-width="1.4"/>')
    add(f'<line x1="{x - 24}" y1="{y - 22}" x2="{x + 24}" y2="{y - 22}" stroke="#000" stroke-width="1.4"/>')
    add(f'<line x1="{x}" y1="{y + 5}" x2="{x - 20}" y2="{y + 38}" stroke="#000" stroke-width="1.4"/>')
    add(f'<line x1="{x}" y1="{y + 5}" x2="{x + 20}" y2="{y + 38}" stroke="#000" stroke-width="1.4"/>')
    for i, line in enumerate(name.split("\n")):
        add(f'<text x="{x}" y="{y + 62 + i * 18}" text-anchor="middle" font-size="15" font-weight="bold">{line}</text>')


for _, y in left:
    link(physician, left_x, y)
for _, y in shared:
    link(physician, mid_x, y)
    link(admin, mid_x, y)
for _, y in right:
    link(admin, right_x, y)
for label, y in left:
    ellipse(left_x, y, label)
for label, y in shared:
    ellipse(mid_x, y, label)
for label, y in right:
    ellipse(right_x, y, label)
actor(*physician, "Physician")
actor(*admin, "Super\nAdministrator")
add("</svg>")

with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(out))
print("wrote", sys.argv[1])
