"""Educational neuro-symbolic pipeline. No clinical use."""
import re
from dataclasses import dataclass, asdict
from functools import lru_cache

TERMS = {
 'chest pain': ('Chest pain','29857009',['chest pain','chest tightness','pressure in my chest']),
 'dyspnea': ('Dyspnea','267036007',['shortness of breath','difficulty breathing','trouble breathing','breathless','cannot breathe','can’t breathe']),
 'headache': ('Headache','25064002',['headache','head pain']),
 'fever': ('Fever','386661006',['fever','high temperature']),
 'abdominal pain': ('Abdominal pain','21522001',['abdominal pain','stomach pain','belly pain']),
 'weakness': ('Weakness',None,['weakness','weak on one side','one-sided weakness']),
 'facial droop': ('Facial weakness',None,['facial droop','face drooping','drooping face']),
 'speech difficulty': ('Speech disturbance',None,['slurred speech','difficulty speaking','cannot speak']),
 'unresponsive': ('Unresponsive',None,['unresponsive','unconscious','not responding']),
 'seizure': ('Seizure',None,['seizure','convulsions']),
 'bleeding': ('Bleeding',None,['bleeding','heavy bleeding']),
 'ankle pain': ('Ankle pain',None,['ankle pain','twisted ankle']),
 'sore throat': ('Sore throat',None,['sore throat']),
 'confusion': ('Confusion',None,['confusion','suddenly confused']),
}
EXAMPLES = {
 'Chest pain + breathing difficulty': {'complaint':'Severe chest pain and shortness of breath','age':45,'spo2':97,'hr':86,'rr':18,'systolic':120,'resources':'Unknown','life_saving':False},
 'Possible stroke': {'complaint':'Sudden facial droop, weakness on one side and slurred speech','age':67,'spo2':97,'hr':92,'rr':19,'systolic':165,'resources':'Unknown','life_saving':False},
 'Low oxygen': {'complaint':'Difficulty breathing with severe breathlessness','age':58,'spo2':85,'hr':120,'rr':30,'systolic':110,'resources':'Unknown','life_saving':False},
 'Immediate intervention': {'complaint':'Patient unresponsive; clinician indicates immediate life-saving intervention','age':72,'spo2':80,'hr':40,'rr':6,'systolic':70,'resources':'Unknown','life_saving':True},
 'Minor ankle injury': {'complaint':'Ankle pain after twisting foot','age':24,'spo2':99,'hr':78,'rr':16,'systolic':118,'resources':'1','life_saving':False},
 'Sore throat': {'complaint':'Sore throat without breathing difficulty','age':20,'spo2':99,'hr':75,'rr':16,'systolic':115,'resources':'0','life_saving':False},
 'Abdominal pain': {'complaint':'Abdominal pain for two days','age':34,'spo2':99,'hr':90,'rr':18,'systolic':122,'resources':'2+','life_saving':False},
 'Unknown symptoms': {'complaint':'I feel strange today','age':35,'spo2':98,'hr':80,'rr':17,'systolic':120,'resources':'Unknown','life_saving':False},
}

def negated(text, start):
    before=text[max(0,start-45):start]
    return bool(re.search(r'\b(?:no|denies|without|not experiencing)\s+(?:(?:any|a|the)\s+)?(?:\w+\s+){0,2}$',before,re.I))

def dictionary_extract(complaint):
    found=[]
    for key,(label,snomed,phrases) in TERMS.items():
        hits=[]
        for phrase in phrases:
            for m in re.finditer(r'(?<!\w)'+re.escape(phrase)+r'(?!\w)',complaint,re.I):
                if not negated(complaint,m.start()):hits.append((m.start(),phrase))
        if hits:
            _,mention=min(hits)
            found.append({'concept_key':key,'mention':mention,'canonical':label,'snomed_id':snomed,'mapping_status':'Illustrative dictionary match' if snomed else 'Unmapped in local dictionary','confidence':None,'extractor':'Dictionary fallback'})
    return found

@lru_cache(maxsize=1)
def load_model():
    from transformers import pipeline
    # Downloads model weights on first use; no medical fine-tuning claimed.
    return pipeline('zero-shot-classification',model='typeform/distilbert-base-uncased-mnli')

def extract(complaint, use_model=False):
    findings=dictionary_extract(complaint)
    info={'mode':'Dictionary fallback','warning':None}
    if not use_model:return findings,info
    try:
        classifier=load_model()
        labels=[TERMS[k][0] for k in TERMS]
        prediction=classifier(complaint,candidate_labels=labels,multi_label=True)
        scores=dict(zip(prediction['labels'],prediction['scores']))
        for item in findings:
            item['confidence']=round(float(scores.get(item['canonical'],0)),3)
            item['extractor']='Pretrained NLI scorer + lexical evidence'
        info['mode']='Pretrained DistilBERT MNLI scoring + lexical evidence'
        info['warning']='Model scores are zero-shot semantic relevance, not calibrated medical probabilities. Lexical evidence is required before accepting findings.'
    except Exception as e:
        info['warning']=f'Pretrained model unavailable; dictionary fallback used ({type(e).__name__}: {str(e)[:180]}).'
    return findings,info

def assess(case, use_model=False):
    complaint=str(case.get('complaint','')).strip()
    findings,extractor=extract(complaint,use_model)
    keys={f['concept_key'] for f in findings}
    trace=[];alerts=[];red_flags=[]
    spo2=case.get('spo2');hr=case.get('hr');rr=case.get('rr');bp=case.get('systolic')
    if 'chest pain' in keys and 'dyspnea' in keys:red_flags.append('Chest pain with breathing difficulty')
    if 'facial droop' in keys or 'speech difficulty' in keys:red_flags.append('Possible acute neurological deficit')
    if 'unresponsive' in keys:red_flags.append('Altered responsiveness')
    if 'seizure' in keys:red_flags.append('Seizure mentioned')
    if 'bleeding' in keys:red_flags.append('Bleeding mentioned; severity needs assessment')
    if spo2 is not None and spo2<90:red_flags.append('Low oxygen saturation (configured threshold)')
    if bp is not None and bp<90:red_flags.append('Low systolic blood pressure (configured threshold)')
    if not complaint:
        level=None;trace.append('R0: Chief complaint missing -> insufficient input.')
    elif case.get('life_saving'):
        level=1;trace.append('R1: Operator explicitly indicated immediate life-saving intervention -> provisional ESI 1.')
    elif red_flags:
        level=2;trace.append('R2: Configured high-risk signal(s) detected -> provisional ESI 2, clinician escalation.')
    else:
        res=case.get('resources','Unknown')
        if res=='Unknown':level=None;trace.append('R3: Expected resources unknown -> ESI 3–5 not determined.')
        elif res=='2+':level=3;trace.append('R4: Two or more anticipated ED resources -> provisional ESI 3.')
        elif res=='1':level=4;trace.append('R5: One anticipated ED resource -> provisional ESI 4.')
        else:level=5;trace.append('R6: No anticipated ED resources -> provisional ESI 5.')
    if red_flags:alerts.extend(red_flags)
    if level==3 and ((hr is not None and hr>100) or (rr is not None and rr>20) or (spo2 is not None and spo2<92)):
        alerts.append('Danger-zone vital signs: higher-acuity review required')
        trace.append('S1: Vital-sign escalation advisory for provisional ESI 3.')
    if not findings:alerts.append('No supported symptom concepts recognized; human review required.')
    if level is None:alerts.append('Insufficient information for a complete provisional ESI classification.')
    trace.append('S2: Educational ESI-inspired subset only; does not implement or validate the complete ESI algorithm.')
    return {'case':dict(case),'esi':level,'findings':findings,'extractor':extractor,'trace':trace,'alerts':alerts,'red_flags':red_flags,'needs_review':bool(alerts) or level is None}
