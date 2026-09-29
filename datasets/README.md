# Datasets

## Fields

The following fields are required for all datasets in this project.

| Field | Type | Description |
| --- | --- | --- |
| id | string or number | Sample identifier |
| question | string | Natural language question |
| sparql | string | Target SPARQL query |
| meta_data | object | Metadata container |
| meta_data.compositionality_type | string | Question compositionality category |
| answer | array[object] | Gold answer list |
| answer[].id | string | Answer entity or value id |
| answer[].name | string | Answer surface name or value |
| topic_entity | array[object] | Mentioned entities in the question |
| topic_entity[].id | string | Topic entity id |
| topic_entity[].name | string | Topic entity name |

## WebQSP answerability groups

`webqsp/webqsp_answerability_groups.csv` (`question_id,group`) places each of
the 1,639 official WebQSP test questions in exactly one group. Rows are sorted
by the numeric part of the QuestionId. Analysis and plotting read this file;
they never re-query the graph.

| group | n | Rule |
|---|---|---|
| `empty_gold` | 11 | no answer on any parse in the official Microsoft release (`webqsp_official_gold.json`, the evaluator's own `empty_gold` rule) |
| `query_mismatch` | 11 | non-empty gold, but the stored gold query (`webqsp.json` `sparql`, i.e. `Parses[0]`) executed unchanged on our Freebase returns none of that parse's gold answers |
| `expected` | 1,617 | non-empty gold, and the stored gold query returns all of it |

Built once by `scripts/build_webqsp_answerability_groups.py` against the local
Virtuoso Freebase endpoint (2026-09-15, ~5 s). The script stops rather than
guess if any question fits no group (query error/timeout, or only part of the
gold returned); none did. `measurement/webqsp_answerability.load_groups`
re-validates the file on every load: 1,639 official ids, no duplicates, counts
11 / 11 / 1,617, and `empty_gold` identical to the official release.

The result matches the earlier read-only answerability audit (2026-09-11) id for
id. Empty gold: WebQTest-238, 353, 475, 722, 863, 897, 974, 1208, 1391, 1600,
1703. Gold-query mismatch (gold MID → MID the stored query returns; 10 of the
11 queries carry a temporal filter):

| question_id | official gold | our Freebase returns |
|---|---|---|
| WebQTest-58 | m.0fbtm7 | m.05fc8c9 |
| WebQTest-65 | m.05wh0sh | m.03_lf |
| WebQTest-671 | m.0357cd | m.021sv1 |
| WebQTest-771 | m.0tc7 | m.03kmb2 |
| WebQTest-800 | m.081pw | m.0gc1_ |
| WebQTest-1088 | m.02k4b2 | m.023v4_ |
| WebQTest-1154 | m.01m4rtc | m.010hn |
| WebQTest-1539 | m.06_bq1 | m.036hf4 |
| WebQTest-1719 | m.0jzkdh | m.01933d |
| WebQTest-1729 | m.01_tz | m.03p77 |
| WebQTest-2019 | m.0hj6dmn | m.055c8 |

The groups describe the dataset against this Freebase instance only; they do
not change scoring. All 1,639 questions stay in the canonical evaluation.
