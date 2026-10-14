# Role-Based Account Administration — Implementation and Evidence Report

- **Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy (DR-CDSS)
- **Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Scope:** two roles (`SUPERADMIN`, `PHYSICIAN`), four account states, Super Administrator provisioning, physician self-registration with approval, mandatory first-sign-in password change, per-physician record separation, administrative audit, migration of the existing database.
- **Status:** implemented and tested by the author; **submitted for independent review, not declared approved.**
- **Date:** 2026-10-13

## 1. Summary of the implemented behaviour

| Rule from the specification | How it is met | Where |
| :--- | :--- | :--- |
| Two roles, four states | `role` ∈ {`SUPERADMIN`, `PHYSICIAN`}; `status` ∈ {`PENDING_APPROVAL`, `ACTIVE`, `REJECTED`, `SUSPENDED`}; only `ACTIVE` can sign in | `backend/app/models/models.py` |
| Super Administrator creates physicians, active at once, temporary password changed at first sign-in | `POST /admin/users` creates an `ACTIVE` account with `must_change_password=true`; the temporary password is returned once; every route but `/auth/me`, `/auth/logout`, `/auth/change-password` answers 403 `PASSWORD_CHANGE_REQUIRED` until it is changed | `account_service.create_physician`, `auth.get_current_physician` |
| Self-registration stays pending | `POST /auth/register` creates `PENDING_APPROVAL`; the request schema has no role or status field; sign-in answers 403 with the state in words | `account_service.register_physician` |
| Only a Super Administrator approves, rejects, suspends, reactivates | every `/admin` route depends on `get_current_superadmin`; the service re-checks the actor | `backend/app/routers/admin.py` |
| Public registration never creates a Super Administrator | the only path to `SUPERADMIN` is `bootstrap_superadmin`, which refuses when one exists | `account_service.bootstrap_superadmin` |
| The last active Super Administrator cannot be suspended | `suspend()` counts active Super Administrators and answers 409 | `account_service.suspend` |
| No shared demonstration account | the seeded `demo.clinician` code is removed; the migration sets any existing row of that name `SUSPENDED`; a gate rule fails if its identifier or password reappears in the application or the interface | `0002_account_administration.py`, `test_no_seeded_demonstration_account_exists` |
| Records belong to the physician who created them | worklist, search and every single-record route take the caller's account id; another account's record is 404 | `assessment_service.get_assessment_by_id(owner_id=...)` |
| Super Administrator is not granted clinical content | `get_current_physician` answers 403 to a `SUPERADMIN`; the console shows per-account counts and audit only | `auth.get_current_physician` |
| Tokens carry only what authorisation needs | claims `sub`, `role`, `tv` (session generation), `iat`, `exp`; `tv` is checked against the account on every request, so password change, suspension, rejection and sign-out invalidate earlier tokens | `auth.login`, `auth._resolve_token` |
| No secret in any response | hashes and token versions are never serialised; a temporary password appears once, in the response that issued it; failed sign-ins are audited without the attempted password | `schemas/auth.py`, `test_13_...` |

## 2. Super Administrator provisioning

- `python backend/scripts/bootstrap_superadmin.py` prompts for username, email and a temporary password (or reads `SUPERADMIN_USERNAME`, `SUPERADMIN_EMAIL`, `SUPERADMIN_FULL_NAME`, `SUPERADMIN_INITIAL_PASSWORD`). The same function runs at application start-up when those variables are set (`backend/main.py`, `docker-compose.yml` passes them through; `.env.example` documents them blank).
- Idempotent: when any Super Administrator exists nothing is created and the command says so (`test_bootstrap_is_idempotent_and_audited`).
- The password is temporary (`must_change_password=true`); the creation writes a `SUPERADMIN_CREATED` audit event.
- No credential is in the source. A gate rule bans the former demonstration identifier and password from the application and interface.

## 3. Database migration

Alembic is introduced (`backend/alembic.ini`, `backend/alembic/env.py`, `backend/alembic/versions/`). `init_db` (`backend/app/core/database.py`) creates missing tables, stamps a database that predates Alembic at `0001_baseline`, and upgrades to `0002_account_administration`:

- `users` gains `status`, `must_change_password`, `token_version`, `created_by_id`, `approved_by_id`, `approved_at`, `rejection_reason`, `last_login_at` (self-referencing foreign keys with `ON DELETE SET NULL`; index on `status`).
- `audit_events` gains `event_type` (indexed), `target_user_id`, `outcome`.
- Backfill: every pre-existing role value becomes `PHYSICIAN` (no existing row becomes a Super Administrator); `status` is set from `is_active`; a row named `demo.clinician` is set `SUSPENDED` with its sessions invalidated. No row is deleted; no assessment, review or report is re-parented.
- Each `ADD COLUMN` is skipped when the column exists, so the revision is safe on a database `create_all` built from the new models, and a second run changes nothing.

Verified on a `pg_dump` copy of the development database (34 assessments): after the first start `alembic_version` = `0002_account_administration`, `demo.clinician` = `PHYSICIAN / SUSPENDED / token_version 1`, the new columns, foreign keys and indexes present, 34 assessments retained; a second start in a separate process reported nothing to change. The development stack itself was then migrated the same way at start-up.

The user record now holds: id, full name, unique username, unique required email, bcrypt hash, role, status, must-change-password flag, creator id, approver id, approval time, last successful sign-in, creation and update times, and an optional professional registration number that is stored as typed and **not verified** against any register.

## 4. Files changed

Backend: `app/models/models.py`, `app/core/config.py`, `app/core/database.py`, `app/core/security.py`, `app/schemas/auth.py`, `app/routers/auth.py` (rewritten), `app/routers/admin.py` (new), `app/routers/assessments.py`, `app/services/account_service.py` (new), `app/services/assessment_service.py`, `main.py`, `alembic.ini` + `alembic/` (new), `scripts/bootstrap_superadmin.py` (new), `scripts/assemble_submission_package.py`, `tests/conftest.py`, `tests/test_account_management.py` (new), `tests/test_api_endpoints.py`, `tests/test_governance_and_security.py`, `tests/test_editor_integrity_gate.py` (three rules rewritten).

Frontend: `src/types/clinical.ts`, `src/services/api.ts`, `src/utils/session.ts` (new, with tests), `src/utils/apiError.test.ts` (new), `src/screens/SignInScreen.tsx`, `src/screens/RegisterScreen.tsx` (new), `src/screens/ChangePasswordScreen.tsx` (new), `src/screens/AdminConsoleScreen.tsx` (new), `src/App.tsx`, `src/components/Header.tsx`, `scripts/capture_account_screens.js` (new), `scripts/capture_live_screenshots.js`, `scripts/capture_two_states.js`, `scripts/audit_mobile_layout.js` (credentials now from the environment; no default), `package.json` (`vitest`, `npm test`), `vitest.config.ts`.

Configuration and documentation: `.env.example`, `docker-compose.yml`, `docs/chapter4/api_contract.md`, `database_schema.md`, `architecture.md` (use-case, state and sequence diagrams), `requirements_test_matrix.md` (FR-11–FR-14, NFR-09–NFR-10), `implementation_status.md`, `system_test_report.md`, `objective_traceability_matrix.md`, `known_limitations.md` (§13), `evidence_provenance.md`, `screenshot_evidence_manifest.md`, `docs/TESTING_RUNBOOK.md` (user manual), `docs/ui_ux_clinical_specification.md`, this report.

## 5. Endpoints

| Route | Who | Introduced or modified |
| :--- | :--- | :--- |
| `POST /api/v1/auth/register` | public | new |
| `POST /api/v1/auth/login` | public | modified: account state, audited failures, claims `sub`/`role`/`tv` |
| `POST /api/v1/auth/logout` | any active account | new: invalidates the account's sessions |
| `GET /api/v1/auth/me` | any active account | modified: role, status, `mustChangePassword` |
| `POST /api/v1/auth/change-password` | any active account | new; mandatory while the password is temporary |
| `GET /api/v1/admin/summary`, `GET /admin/users`, `GET /admin/users/pending`, `GET /admin/users/{id}` | SUPERADMIN | new |
| `POST /api/v1/admin/users`, `POST /admin/users/batch` | SUPERADMIN | new (temporary password returned once) |
| `PATCH /api/v1/admin/users/{id}/approve` · `/reject` · `/suspend` · `/reactivate` | SUPERADMIN | new |
| `POST /api/v1/admin/users/{id}/reset-password` | SUPERADMIN | new |
| `GET /api/v1/admin/audit-events` | SUPERADMIN | new, read-only |
| every `/api/v1/assessments...` and `/reports/...` route | PHYSICIAN, own records | modified: `get_current_physician`, owner scoping |

Full shapes: `docs/chapter4/api_contract.md` §1 and §1b.

## 6. Role and status transition rules

```
self-registration ──► PENDING_APPROVAL ──approve──► ACTIVE ◄──reactivate── SUSPENDED
                            │                          │
                          reject                    suspend (not the last active SUPERADMIN)
                            ▼                          ▼
                         REJECTED                   SUSPENDED
Super Administrator creates ──► ACTIVE (must_change_password = true)
```

A transition from any other state answers 409. Approve, reject, suspend and reactivate require an active Super Administrator; an account cannot approve or reject itself. Rejection, suspension, password change, password reset and sign-out increment `token_version`.

## 7. Tests added

`backend/tests/test_account_management.py` — 25 tests, numbered after the sixteen required cases: 01 create, 02 active + mandatory change, 03/03b pending and no role through a public request, 04 pending locked out, 05 approval, 06 rejection, 07 suspension, 08 reactivation, 09 physician vs `/admin`, 10/10b/10c role through a request, Super Administrator not granted clinical content, no self-approval, 11 record separation, 12 empty history, 13 no secrets, 14/14b audit (administrative events; review confirmation), 15 last Super Administrator; plus bootstrap idempotence, batch creation, token claims, sign-out, expired/tampered tokens, registration input rules. Case 16 (existing tests keep passing) is the full suite below.

Fixtures (`tests/conftest.py`) bootstrap a Super Administrator per test through the same service a deployment uses and create physicians through the API; the database is dropped after every test.

Frontend: `src/utils/session.test.ts` (routing by role and temporary password, status messages, password policy, registration validation with no role field) and `src/utils/apiError.test.ts` — 14 tests under `vitest`.

## 8. Exact test results

Every figure below is read from a recorded run of a fresh extraction of the Chapter Four evidence archive, not from a development run:

- Backend, complete suite, `python -m pytest tests/ -v` in a new virtual environment built from `requirements.txt`: **312 collected — 311 passed, 0 failed, 1 skipped**, as recorded in [`docs/chapter4/test_execution.log`](chapter4/test_execution.log) (environment: [`test_environment_freeze.txt`](chapter4/test_environment_freeze.txt)). The skip is the pre-existing conditional `test_contaminated_results_are_labelled_superseded`. The suite has grown since this feature was first measured (295 collected): the probability-calibration suite and the session-teardown test were merged afterwards.
- Backend, `tests/test_account_management.py`, within that run: **25 passed, 0 failed**.
- Frontend, from the same extraction and recorded in [`docs/chapter4/build_verification.log`](chapter4/build_verification.log): `npx tsc --noEmit` clean, `npm test` (vitest) **2 files, 14 tests passed**, and `npm run build` succeeds.
- Mutation-style checks performed while writing the rules: the demonstration-account rule was confirmed to fail on the pre-change `auth.py`.

## 9. Screenshots

`docs/chapter4/screenshots/accounts/01_sign_in.png` … `10_account_audit_history.png`, captured against the running development stack by `frontend/scripts/capture_account_screens.js` on the date in `capture_record.json`, which also names the two accounts the capture created. Frames and captions: `docs/chapter4/screenshot_evidence_manifest.md` §1b. The Super Administrator used for the capture was bootstrapped on the development database; its password is not in the repository.

## 10. Known limitations

- Professional registration numbers are stored as typed; nothing verifies them (interface and documents say so).
- No email is sent; approvals and temporary passwords are communicated outside the system.
- Record separation is by creating account; there is no transfer of records between accounts, and the former demonstration account's 34 records are visible to no physician.
- The image and heatmap file routes remain unauthenticated (`known_limitations.md` §12); the API refuses another physician's record, but the files behind it are served to anyone holding the filename.
- Session invalidation is by token version checked on each request; there is no revocation list.
- No rate limiting on sign-in.
- The bootstrap password travels through the environment or a prompt; the deployment must protect its environment file.

## 11. Remaining work

- Deploy: the production database at `/opt/cdss` will be migrated by `init_db` at the next start; a Super Administrator must then be bootstrapped there (environment or command) before anyone can sign in, because the former demonstration account is suspended by the migration.
- Independent review of this report and of the evidence above.
