# Evaluation procedure

The Evaluate page tests text-to-image semantic retrieval without manual-tag prioritization or filters. Select every relevant indexed filename before running a query, including duplicate copies. The application retrieves up to five results.

- Precision@5 = relevant retrieved results / 5. The denominator stays five even if fewer results are available.
- Recall@5 = relevant retrieved results / all marked relevant filenames.
- Search time includes query encoding and database retrieval, excluding model startup, manual annotation, and rendering.

Runs append to `evaluations.jsonl` in the local project folder. The records contain the query, corpus filenames, annotations, model identifier, dimensions, ranked results, metrics, and timing. The corpus token detects filename changes, not changes to embeddings.

The five-file butterfly demonstration contains duplicated images. It must not be described as 100% overall accuracy or evidence of general performance.

For a stronger future evaluation, use a diverse collection with duplicate content removed, write multiple queries before inspecting results, label all relevant images, and report the number of queries and images alongside mean Precision@5 and Recall@5. Report latency across repeated runs, separating cold startup from warmed search. Keep tuning queries separate from final evaluation queries.
