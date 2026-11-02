# QueryLane Evaluation Dashboard

> [!IMPORTANT]
> **Gate Status: **FAILED****  
> Overall Accuracy: **75.0%** (30/40 passed) | Fireworks Tokens: **3,000** across **40** calls (avg **75.0** tok/task)

## Per-Category Performance Breakdown

| Category | Tasks | Passed | Failed | Accuracy | Gate (80%) | Fireworks Tokens | API Calls | Avg Tok/Task | Routed Lanes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **factual** | 5 | 4 | 1 | 80.0% | PASS | 375 | 5 | 75.0 | `cloud`: 5 |
| **math** | 5 | 5 | 0 | 100.0% | PASS | 375 | 5 | 75.0 | `cloud`: 5 |
| **sentiment** | 5 | 5 | 0 | 100.0% | PASS | 375 | 5 | 75.0 | `cloud`: 5 |
| **summarization** | 5 | 3 | 2 | 60.0% | FAIL | 375 | 5 | 75.0 | `cloud`: 5 |
| **ner** | 5 | 5 | 0 | 100.0% | PASS | 375 | 5 | 75.0 | `cloud`: 5 |
| **code_debug** | 5 | 3 | 2 | 60.0% | FAIL | 375 | 5 | 75.0 | `cloud`: 5 |
| **logic** | 5 | 5 | 0 | 100.0% | PASS | 375 | 5 | 75.0 | `cloud`: 5 |
| **code_gen** | 5 | 0 | 5 | 0.0% | FAIL | 375 | 5 | 75.0 | `cloud`: 5 |
| **TOTAL** | **40** | **30** | **10** | **75.0%** | ****FAILED**** | **3,000** | **40** | **75.0** | - |

## Lane Distribution

| Lane | Task Count | Percentage |
| :--- | :---: | :---: |
| `cloud` | 40 | 100.0% |