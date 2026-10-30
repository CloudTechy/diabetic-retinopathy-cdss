# DR-CDSS design diagrams

Four diagrams of the implemented AI-based Clinical Decision Support System for diabetic retinopathy. Each is drawn from the code, not from a design document: the class and database diagrams follow `backend/app/models/models.py` and the services in `backend/app/services/`, and the sequence diagram follows the API routes and `AssessmentService`.

| File | Diagram | Based on |
| :--- | :--- | :--- |
| `use_case_diagram.png` / `.svg` | Use case diagram | the two actors and what each may do: the Physician (request an account, worklist, create an assessment, upload the fundus image, view validation results, calibrated class probabilities and the Grad-CAM explanation, submit the professional review, view the completed record, search history, view the audit trail, download the PDF report); the Super Administrator (account summary, approve or reject registrations, create physician accounts, suspend or reactivate accounts, reset passwords, view the account audit log); both log in/out and change their password. The Super Administrator has no access to clinical content |
| `class_diagram.png` / `.svg` | Class diagram | the user roles; Assessment and its image, validation result, model execution, AI result, Grad-CAM artifact, professional review and audit events; the validation pipeline and its three gates; the inference engine (the abstract base and its one implementation, EfficientNet-B0, which loads the temperature-scaling calibration artifact); the assessment, report and account services |
| `sequence_diagram.png` / `.svg` | Sequence diagram | sign-in (refusal, and the forced change of a temporary password), creating an assessment and uploading the fundus image, the three fail-closed validation gates, inference with temperature scaling and Grad-CAM, the fail-closed path when the checkpoint or calibration artifact cannot be loaded or verified (503, no result saved), the professional review rule (justification required to disagree or for an inconclusive response), the write-locked record, and the PDF report |
| `database_schema.png` / `.svg` | Database schema (entity-relationship) | all 10 tables with every column, type and key (PK, FK, UK) and their cardinalities |

- **PNG** files are rendered at 2× scale for documents and slides.
- **SVG** files scale without loss.
- `source/*.mmd` are the Mermaid sources of the class, sequence and schema diagrams; `source/use_case_diagram.py` generates the use case diagram (Mermaid has no use-case diagram type). Each can be edited and re-rendered.

## Notes for the thesis

- **Account roles:** Physician and Super Administrator are the two values of `users.role`, shown as specialisations of User.
- **Cardinalities:** one-to-one relationships (an assessment has at most one image, validation result, model execution, AI result and professional review) are enforced by unique foreign keys. An AI result can have several explanation artifacts, and an assessment has many audit events.
- **Calibration:** the temperature (T = 1.0939) is fitted on the validation partition and applied to the logits before the class probabilities are reported. It is stored in the calibration artifact, not in the database. `AIResult.classScores` (`ai_results.class_scores`) holds the five temperature-scaled class probabilities.
- **One inference engine:** the application serves only the fixed EfficientNet-B0 checkpoint (SHA-256 `67d0b896…3dbf7`). There is no simulated engine and no setting that selects one; if the checkpoint or calibration artifact cannot load, processing fails explicitly. The automated tests use a test double kept in `backend/tests/`, which is not part of the application and is not drawn.
