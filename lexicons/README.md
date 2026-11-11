# Lexicons

| file | role | masked in |
|---|---|---|
| topic.txt | words that name depression, its diagnosis or treatment (suspected shortcuts) | mask-topic, mask-topic+sui |
| suicide.txt | explicit suicide / self-harm terms | mask-topic+sui |
| symptom.txt | PHQ-9 / DSM-5 symptom language (8 domains; item 9 = suicide.txt) | never (attribution analysis only) |

Compiled from the PHQ-9 items, the DSM-5 criteria for a major depressive episode, and common
antidepressant generic and brand names. The first draft was written with AI assistance and then
reviewed and approved by the authors (see [AI_DISCLOSURE.md](../AI_DISCLOSURE.md)). The lists were
fixed on 2026-11-08, before any masked data was created.

Prevalence per class: `results/data_reports/lexicon_preview.md`.
