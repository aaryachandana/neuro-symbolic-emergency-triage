# NeuroTriage AI — Enhanced Academic Prototype

**Neuro-Symbolic AI Framework for Explainable Emergency Triage**

An academic research demonstration only. **Not for clinical use or real patient decisions.** The triage algorithm is an illustrative ESI-inspired subset, not the full validated Emergency Severity Index algorithm.

## Run on Windows (VS Code terminal)

1. Extract this ZIP, open the `neurotriage-final-project` folder in VS Code.
2. Run `python -m pip install -r requirements.txt`.
3. Run `python -m streamlit run app.py`.
4. Open the local address shown, normally `http://localhost:8501`.

If another Streamlit instance is already running, stop it using Ctrl+C in its terminal first.

## Optional pretrained neural component

For the **optional** pretrained model, install `python -m pip install -r requirements-ai.txt` and switch on **Enable pretrained AI scorer** in the app sidebar. On first use, the application downloads `typeform/distilbert-base-uncased-mnli` from Hugging Face (internet required). The model is a general-domain NLI model, **not** medically trained or validated; its scores are not clinical probabilities. Accepted concepts still require lexical evidence. If unavailable, the app falls back to its dictionary extractor and shows a warning. Default mode works offline.

## Modules

- `app.py`: Streamlit dashboard, custom input, explanation, evaluation, and input snapshots.
- `modules/engine.py`: lexical evidence, optional pretrained scoring, illustrative concept IDs, simplified ESI rules, alerts, and trace.
- `modules/legacy_core.py`: original starter logic for comparison only; not used by app.
- `tests/test_engine.py`: reproducible tests.
- `data/`: sample evaluation CSV.

## Evaluation integrity

The built-in synthetic scenario agreement measures only whether **eight designed examples** agree with expected demonstration outputs. It is **not clinical accuracy**. The optional evaluation CSV computes agreement against user-provided labels; meaningful accuracy requires independent high-quality reference labels. Do not present the synthetic examples as MIMIC-IV-ED results or report hybrid vs. LLM-only comparisons that have not been performed.

## Current limitations

1. Pretrained model is optional and general domain; no medical NER model is bundled.
2. SNOMED identifiers are local illustrative entries, not verified by a live terminology service.
3. ESI logic is intentionally simplified and incomplete.
4. Red-flag keyword and threshold coverage is limited; misses are possible.
5. No MIMIC-IV-ED data or clinically validated evaluation is included.

For the report, describe these as implemented prototype capabilities and future enhancements, not as completed clinical validation.
