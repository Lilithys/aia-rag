# Case Study – RAG + Generative AI Service
Role: Junior Developer
## Unified Business Scenario & Global Constraints
Build a retrieval‑augmented QA system on "~/doc". The system must support multi‑turn dialogue and grounded, citation‑backed answers.
Candidates may choose any tech stack. Key technical choices must be justified in the deliverables with clear, quantitative evidence.
## Global Constraints (apply unless overridden below)
1.	API & Response: accept natural‑language questions and return answers grounded in retrieved content.
2.	Latency: 90% of user questions must return within 10 seconds end‑to‑end.
3.	Cost: provide token‑cost estimate per 1,000 calls and a sensitivity analysis for top_k, reranker on/off, and temperature (≥ 3 settings).
4.	Quality Metrics:
•	RAG: Faithfulness ≥ 0.85 (define your rubric) and Context Precision ≥ 0.70.
•	Logging & Tracing: design a logging system suitable for generation monitoring and issue diagnosis.
•	Security: minimal prompt‑injection defense; basic PII handling; answers must strictly rely on retrieved context.
## Objective
Deliver a minimum viable RAG‑based QA system with basic multi‑turn context, citation‑backed answers, and foundational logging that enables issue diagnosis and iteration.
## Functional Requirements
1.	1) RAG QA
•	Provide QA over internal documents; answers must leverage retrieved sources and include citations.
2.	2) Multi‑turn Dialogue
•	Support conversation continuity within the same session without requiring users to repeat prior context.
Non‑Functional & Quantitative Metrics
1.	1) Correctness & Quality
•	Overall answer accuracy ≥ 80% on your own evaluation set; meet the Faithfulness and Context Precision thresholds above.
2.	2) Evolvability
•	Design for iterative improvements (scaling documents, retrieval/generation strategy tweaks, metric/logging enhancements).
3.	3) Logging & Observability
•	Logs must be adequate for performance analysis and generation‑quality troubleshooting (include sample logs).
4.	4) Robustness
•	When retrieval returns no results or very low similarity, return an explanatory refusal and clearly communicate current capability boundaries.
## Deliverables
• A runnable demo (local or cloud).
• Complete source code and README (run/config).
• A 200–500 word design note explaining key choices and trade‑offs.
• An evaluation summary with metric tables and PII‑redacted sample logs.
