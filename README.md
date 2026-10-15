# NeuroTriage AI

**Neuro-Symbolic AI Framework for Explainable Emergency Triage**

Academic final-year project prototype. **Not for clinical use or real patient decisions.**

## Overview

A Streamlit demonstration combining optional pretrained DistilBERT MNLI scoring with lexical evidence, simplified ESI-inspired symbolic rules, illustrative ontology mapping, red-flag alerts, and an explanation/proof trace.

## Setup

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-ai.txt
python -m streamlit run app.py
```

Enable the pretrained AI scorer in the Streamlit sidebar. The optional general-domain model downloads from Hugging Face on first use and is not medically validated.

## Limitations

- ESI rules are illustrative and incomplete.
- SNOMED concept IDs are illustrative mappings, not verified terminology-service matches.
- The pretrained model is general-domain, not a clinically validated medical extractor.
- Synthetic-case agreement is not clinical accuracy.
- No restricted patient records should be uploaded to this public repository.

Source files and tests will be added separately.
