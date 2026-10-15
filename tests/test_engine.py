from modules.engine import assess, EXAMPLES, dictionary_extract

def test_synthetic_examples():
    expected={'Chest pain + breathing difficulty':2,'Possible stroke':2,'Low oxygen':2,'Immediate intervention':1,'Minor ankle injury':4,'Sore throat':5,'Abdominal pain':3,'Unknown symptoms':None}
    for name,level in expected.items():
        assert assess(EXAMPLES[name])['esi']==level,name

def test_esi_one_requires_operator_intervention():
    case=dict(EXAMPLES['Chest pain + breathing difficulty'])
    assert assess(case)['esi']==2
    case['life_saving']=True
    assert assess(case)['esi']==1

def test_negation():
    assert not any(f['concept_key']=='dyspnea' for f in dictionary_extract('No shortness of breath'))

def test_snapshot():
    case=dict(EXAMPLES['Possible stroke']);r=assess(case);case['complaint']='changed'
    assert r['case']['complaint']!=case['complaint']
