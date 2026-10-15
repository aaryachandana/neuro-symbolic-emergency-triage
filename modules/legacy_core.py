import re
from dataclasses import dataclass, asdict

CONCEPTS = {
    'chest pain': ('Chest pain', '29857009'),
    'shortness of breath': ('Dyspnea', '267036007'),
    'difficulty breathing': ('Dyspnea', '267036007'),
    'unresponsive': ('Unresponsive', None),
    'unconscious': ('Unconscious', None),
    'stroke': ('Suspected stroke', None),
    'facial droop': ('Facial weakness', None),
    'weakness': ('Weakness', None),
    'bleeding': ('Bleeding', None),
    'ankle pain': ('Ankle pain', None),
    'fever': ('Fever', '386661006'),
    'headache': ('Headache', '25064002'),
    'abdominal pain': ('Abdominal pain', '21522001'),
    'sore throat': ('Sore throat', None),
}
NEGATION = re.compile(r'\b(no|denies|without|not experiencing)\s+(?:any\s+)?$', re.I)


def extract_findings(complaint):
    """Deterministic demonstration fallback, NOT a pretrained LLM."""
    text = complaint.lower()
    findings = []
    for phrase, (canonical, concept_id) in CONCEPTS.items():
        for match in re.finditer(r'\b' + re.escape(phrase) + r'\b', text):
            preceding = text[max(0, match.start()-35):match.start()]
            if NEGATION.search(preceding):
                continue
            findings.append({'mention': phrase, 'canonical': canonical, 'snomed_id': concept_id, 'mapping_status': 'illustrative dictionary mapping' if concept_id else 'unmapped'})
            break
    return findings


def assess(complaint, age=None, spo2=None, hr=None, rr=None, systolic=None, resources=None, life_saving=False):
    findings = extract_findings(complaint)
    terms = {f['mention'] for f in findings}
    trace = []
    alerts = []
    if life_saving:
        level = 1
        trace.append('R1: Immediate life-saving intervention indicated by operator -> ESI 1.')
    elif ('unresponsive' in terms or 'unconscious' in terms):
        level = 2
        trace.append('R2: Altered responsiveness -> high-risk flag and urgent clinician review; provisional ESI 2.')
        alerts.append('Assess immediately for need for life-saving intervention (possible ESI 1).')
    elif (spo2 is not None and spo2 < 90) or ('chest pain' in terms and ('shortness of breath' in terms or 'difficulty breathing' in terms)) or 'facial droop' in terms or 'stroke' in terms:
        level = 2
        trace.append('R3: High-risk presentation -> provisional ESI 2 and immediate clinician review.')
    elif resources is None:
        level = None
        trace.append('R4: Expected ED resources not provided -> cannot safely determine ESI 3–5.')
        alerts.append('Missing resource estimate: clinician review required.')
    elif resources >= 2:
        level = 3
        trace.append('R5: Two or more expected ED resources -> provisional ESI 3.')
    elif resources == 1:
        level = 4
        trace.append('R6: One expected ED resource -> provisional ESI 4.')
    else:
        level = 5
        trace.append('R7: No expected ED resources -> provisional ESI 5.')
    if spo2 is not None and spo2 < 90:
        alerts.append('Low oxygen saturation: urgent assessment required.')
    if level == 3 and ((hr is not None and hr > 100) or (rr is not None and rr > 20) or (spo2 is not None and spo2 < 92)):
        alerts.append('Abnormal vital signs: ESI 3 requires clinician consideration of higher acuity.')
        trace.append('S1: Abnormal vital signs -> escalation for clinician review; no automatic downgrade.')
    if not complaint.strip():
        alerts.append('Missing chief complaint: assessment is incomplete.')
        level = None
    if not findings:
        alerts.append('No known clinical concepts extracted; free text needs human interpretation.')
    trace.append('S2: All outputs are provisional; clinical judgment and full ESI algorithm required.')
    return {'esi': level, 'findings': findings, 'trace': trace, 'alerts': alerts, 'method': 'Dictionary-based demonstration extraction + simplified ESI-inspired rules'}
