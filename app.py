import streamlit as st
import pandas as pd
from modules.engine import assess, EXAMPLES

st.set_page_config(page_title='NeuroTriage AI | Research Prototype',page_icon='🧠',layout='wide')
st.markdown('''<style>.block-container{padding-top:1.4rem}.stMetric{background:#182536;padding:16px;border-radius:12px;border:1px solid #3b526a}h1,h2,h3{letter-spacing:-.025em}</style>''',unsafe_allow_html=True)
st.title('🧠 NeuroTriage AI')
st.caption('Neuro-Symbolic AI Framework for Explainable Emergency Triage | Final-year research prototype')
st.warning('ACADEMIC DEMONSTRATION ONLY — Not for clinical decisions. ESI-inspired subset, illustrative terminology dictionary, and optional general-domain pretrained model.')

if 'case' not in st.session_state:st.session_state.case=dict(EXAMPLES['Chest pain + breathing difficulty'])
if 'selected' not in st.session_state:st.session_state.selected='Chest pain + breathing difficulty'
if 'last' not in st.session_state:st.session_state.last=None
if 'history' not in st.session_state:st.session_state.history=[]

def choose_example():
    selected=st.session_state.example_picker
    if selected!='Custom input':
        st.session_state.case=dict(EXAMPLES[selected]);st.session_state.selected=selected
        for k,v in st.session_state.case.items():st.session_state['input_'+k]=v
    else:st.session_state.selected='Custom input'

def case_widget(k,default):
    key='input_'+k
    if key not in st.session_state:st.session_state[key]=default

for k,v in st.session_state.case.items():case_widget(k,v)
page=st.sidebar.radio('Navigation',['Patient Assessment','Extraction & Ontology','Safety & Proof Trace','Evaluation Dashboard','About the Architecture'])
st.sidebar.divider()
use_model=st.sidebar.toggle('Enable pretrained AI scorer',value=False,help='Downloads a general-domain DistilBERT MNLI model on first use; requires internet, transformers, torch and additional memory. Lexical evidence remains mandatory.')
st.sidebar.caption('Default mode runs fully offline. AI scoring is optional, not a medically trained extractor.')
st.sidebar.info('Use fictional patient descriptions only; do not enter identifiable patient data.')

if page=='Patient Assessment':
    st.header('Patient Assessment')
    st.selectbox('Load a synthetic example',list(EXAMPLES)+['Custom input'],index=list(EXAMPLES).index(st.session_state.selected) if st.session_state.selected in EXAMPLES else len(EXAMPLES),key='example_picker',on_change=choose_example)
    st.text_area('Chief complaint — freely editable',key='input_complaint',height=110)
    a,b,c=st.columns(3)
    with a:
        st.number_input('Age (years)',min_value=0,max_value=120,key='input_age')
        st.number_input('SpO₂ (%)',min_value=0,max_value=100,key='input_spo2')
    with b:
        st.number_input('Heart rate (bpm)',min_value=0,max_value=250,key='input_hr')
        st.number_input('Respiratory rate (/min)',min_value=0,max_value=80,key='input_rr')
    with c:
        st.number_input('Systolic BP (mmHg)',min_value=0,max_value=300,key='input_systolic')
        st.selectbox('Expected ED resources', ['Unknown','0','1','2+'],key='input_resources')
    st.checkbox('Immediate life-saving intervention needed (operator-assessed)',key='input_life_saving')
    if st.button('🔎 Analyze Patient Case',type='primary',use_container_width=True):
        case={k:st.session_state['input_'+k] for k in EXAMPLES['Chest pain + breathing difficulty']}
        with st.spinner('Running extraction, mapping, reasoning and safety checks...'):
            output=assess(case,use_model)
        st.session_state.last=output
        st.session_state.history.append({'complaint':case['complaint'],'esi':output['esi'],'red_flag_count':len(output['red_flags']),'extractor':output['extractor']['mode']})
    if st.session_state.last:
        r=st.session_state.last
        st.divider()
        x,y,z=st.columns(3)
        x.metric('Provisional ESI',r['esi'] if r['esi'] is not None else 'Review needed')
        y.metric('Extracted findings',len(r['findings']))
        z.metric('Configured red flags',len(r['red_flags']))
        if r['alerts']:st.error('Safety / review alerts: '+' | '.join(r['alerts']))
        else:st.info('No configured red flag triggered. This does not establish safety.')
        st.subheader('Why did the system choose this result?')
        st.write('**Clinical findings used:** ' + (', '.join(dict.fromkeys(str(f.get('canonical', f.get('mention', 'unknown'))) for f in r['findings'])) if r['findings'] else 'No supported symptom concepts identified'))
        if r['esi'] == 1:
            st.write('**Decision:** The operator marked immediate life-saving intervention as needed. This activates the highest-priority R1 rule.')
        elif r['esi'] == 2:
            st.write('**Decision:** One or more configured high-risk signals activated rule R2. High-risk findings take priority over resource estimates.')
            st.write('**Why not ESI 1?** The operator did not mark immediate life-saving intervention as needed.')
        elif r['esi'] in (3,4,5):
            st.write('**Decision:** No configured high-risk signal was found; the resource estimate determined the provisional level.')
        else:
            st.write('**Decision:** The system lacks enough information to assign a provisional level. Human review is needed.')
        if r['red_flags']:
            st.write('**Red flags:** ' + '; '.join(map(str,r['red_flags'])))
        st.write('**Complete rule-by-rule proof trace:**')
        for n, step in enumerate(r['trace'], 1):
            st.write(f'{n}. {step}')
        st.caption('Explanation reflects implemented educational rules, not a medical diagnosis or validated clinical decision.')
        st.subheader('Input snapshot used for this result')
        st.json(r['case'],expanded=False)
        st.caption('The snapshot stays attached to the result even if you edit or switch the form. Click Analyze again to refresh.')
        st.info('Use the sidebar to inspect ontology mapping and the complete reasoning proof trace.')

elif page=='Extraction & Ontology':
    st.header('Clinical Extraction and Ontology Mapping')
    r=st.session_state.last
    if not r:st.info('Analyze a case first on Patient Assessment.')
    else:
        st.write('**Analyzed complaint:**',r['case']['complaint'])
        st.write('**Extractor:**',r['extractor']['mode'])
        if r['extractor']['warning']:st.warning(r['extractor']['warning'])
        if r['findings']:
            st.dataframe(pd.DataFrame(r['findings'])[['mention','canonical','snomed_id','mapping_status','confidence','extractor']],hide_index=True,use_container_width=True)
        else:st.warning('No known concepts extracted.')
        st.info('SNOMED CT IDs shown here are locally curated illustrative mappings, NOT verified live against a licensed terminology service. Missing IDs are explicitly marked unmapped.')

elif page=='Safety & Proof Trace':
    st.header('Symbolic Reasoning & Safety Proof Trace')
    r=st.session_state.last
    if not r:st.info('Analyze a case first on Patient Assessment.')
    else:
        st.write('**Analyzed complaint:**',r['case']['complaint'])
        st.metric('Provisional ESI',r['esi'] if r['esi'] is not None else 'Review needed')
        st.subheader('Rule execution')
        for n,step in enumerate(r['trace'],1):st.write(f'**{n}.** {step}')
        st.subheader('Red-flag / safety alerts')
        if r['alerts']:
            for alert in r['alerts']:st.warning(alert)
        else:st.info('No configured alerts. Not a safety guarantee.')
        st.subheader('Audit input')
        st.json(r['case'],expanded=False)

elif page=='Evaluation Dashboard':
    st.header('Synthetic Scenario Evaluation')
    st.caption('Reproducible scenario checks against hand-authored demonstration expectations. NOT clinical accuracy or independent validation.')
    expected={'Chest pain + breathing difficulty':2,'Possible stroke':2,'Low oxygen':2,'Immediate intervention':1,'Minor ankle injury':4,'Sore throat':5,'Abdominal pain':3,'Unknown symptoms':None}
    rows=[]
    for name,case in EXAMPLES.items():
        r=assess(case,False)
        rows.append({'scenario':name,'expected':str(expected[name]) if expected[name] is not None else 'Review','observed':str(r['esi']) if r['esi'] is not None else 'Review','match':r['esi']==expected[name],'red_flags':len(r['red_flags'])})
    df=pd.DataFrame(rows)
    a,b,c=st.columns(3)
    a.metric('Synthetic scenarios',len(df));b.metric('Scenario agreement',f"{df['match'].sum()} / {len(df)}");c.metric('Configured red-flag scenarios',int((df['red_flags']>0).sum()))
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.subheader('Scenario outputs')
    st.bar_chart(df.set_index('scenario')['red_flags'])
    st.subheader('Demonstration confusion matrix')
    matrix=pd.crosstab(df['expected'],df['observed'],rownames=['Expected'],colnames=['Observed'])
    st.dataframe(matrix,use_container_width=True)
    st.download_button('Download synthetic scenario results (CSV)',df.to_csv(index=False).encode('utf-8'),'synthetic_scenario_results.csv','text/csv')
    st.divider()
    st.subheader('Optional independent evaluation CSV')
    st.write('Upload a **synthetic, independently labeled** CSV with columns `complaint,age,spo2,hr,rr,systolic,resources,life_saving,reference_esi`. Results use the rule-based extractor for reproducibility.')
    file=st.file_uploader('Upload evaluation CSV',type=['csv'])
    if file:
        try:
            data=pd.read_csv(file)
            required=['complaint','age','spo2','hr','rr','systolic','resources','life_saving','reference_esi']
            missing=[x for x in required if x not in data.columns]
            if missing:st.error('Missing columns: '+', '.join(missing))
            else:
                outcomes=[]
                for _,item in data.iterrows():
                    case={k:(None if pd.isna(item[k]) else item[k]) for k in required if k!='reference_esi'}
                    case['resources']=str(case['resources']) if case['resources'] is not None else 'Unknown'
                    case['life_saving']=str(case['life_saving']).lower() in ('true','1','yes')
                    result=assess(case,False)
                    ref=None if pd.isna(item['reference_esi']) else int(item['reference_esi'])
                    outcomes.append({'reference_esi':ref,'predicted_esi':result['esi'],'match':ref==result['esi']})
                out=pd.DataFrame(outcomes)
                st.metric('Agreement with uploaded labels',f"{out['match'].mean()*100:.1f}%")
                st.dataframe(pd.crosstab(out['reference_esi'].fillna('Review'),out['predicted_esi'].fillna('Review')),use_container_width=True)
                st.download_button('Download predictions',out.to_csv(index=False),'evaluation_predictions.csv','text/csv')
                st.caption('Interpret this metric only in light of how reference labels were obtained; it is not automatically clinical accuracy.')
        except Exception as exc:st.error(f'Unable to process CSV: {exc}')

else:
    st.header('Architecture & Limitations')
    for title,description in [
        ('1. Patient input handler','Editable free-text complaint, age, vital signs and resource estimate; consistent input snapshot.'),
        ('2. Neural extractor','Optional pretrained general-domain DistilBERT MNLI zero-shot relevance scoring. Deterministic lexical evidence is required; this is not a medical NER model.'),
        ('3. Ontology mapper','Local illustrative medical terminology table with SNOMED CT concept IDs where provided. No live terminology service.'),
        ('4. Symbolic reasoner','Auditable, simplified ESI-inspired rules. Only explicit operator-assessed life-saving intervention produces ESI 1.'),
        ('5. Explainability','Numbered triggered-rule proof trace and patient-input snapshot.'),
        ('6. Safety module','Configured red flags, escalation messages, and incomplete-input handling.'),
        ('7. Evaluation','Built-in synthetic scenario concordance and optional independently labeled CSV comparison.')]:
        with st.expander(title,expanded=True):st.write(description)
    st.warning('Future work: clinically reviewed rules, licensed terminology-service integration, medical-domain NER, ESI algorithm validation, MIMIC-IV-ED evaluation with proper access, and comparative baselines.')
