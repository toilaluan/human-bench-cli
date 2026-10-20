# Prompt catalog v1

These 160 English prompts are authored for this project using the eight template
families introduced by Anubhooti Jain, Mayank Vatsa, and Richa Singh in
[HumanBench: Two Heads, No Legs, But Mostly Human, the State of Generative
Capabilities in T2I Models, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Jain_HumanBench_Two_Heads_No_Legs_But_Mostly_Human_the_State_WACV_2026_paper.html).
The authors' [published catalog](https://github.com/IAB-IITJ/HumanBench/blob/main/spat-prompts.csv)
informs the template taxonomy and slot vocabulary. We do not reproduce their
evaluation code or generated image dataset.

Changes include explicit counts, adult subjects, material targets, actor/action
bindings, viewer-relative left/right, and concrete wording for playing and spatial
contexts. The difficulty labels are structural design categories, not measured
human difficulty. Four independent generations per prompt form the standard run.

The catalog contains 20 prompts per family. It includes counts from 2–7 in T6/T7,
and compound groups totaling 4–10 in T8. T3 intentionally includes human-shaped
figures made from materials; anatomical assessment concerns their depicted form,
not living skin or photorealism.

Each counterfactual changes one designated slot. It is a contrast candidate, not
necessarily a logically exclusive scene: two actions or materials may coexist.
Each negation denies the designated required fact. Contrasts are diagnostic texts;
the benchmark does not require images generated from them.

Evaluation reads the committed JSONL directly. `scripts/build_catalog.py --check`
verifies the authoring source against it. Released IDs must never be repurposed;
future catalog changes need a new version. Run metadata stores the complete
catalog snapshot and its canonical SHA-256 hash.

This is an adapted benchmark. Its scores are not numerically interchangeable with
the original HumanBench results.
