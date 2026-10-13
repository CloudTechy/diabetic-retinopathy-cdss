"""
Role-based account administration: Super Administrator and Physician roles,
the four account states, the transitions between them, and the separation of
each physician's records.

The sixteen cases required of this feature are numbered in the test names.
Every account used here is created the way a deployment creates it: the
Super Administrator by the bootstrap service, physicians through the
administration API. Nothing is seeded by the application.
"""
import base64

import pytest

from tests.conftest import (
    SUPERADMIN_CREDENTIALS,
    activated_physician,
    bearer,
    change_password,
    create_physician,
    create_synthetic_retinal_fundus,
    image_to_bytes,
    sign_in,
)

API = "/api/v1"


def register(client, username, password="Self-Reg-Pass-2026", **overrides):
    payload = {"fullName": f"Applicant {username}", "username": username, "email": f"{username}@example.org",
               "password": password, "passwordConfirmation": password}
    payload.update(overrides)
    return client.post(f"{API}/auth/register", json=payload)


def image_data_url():
    img = create_synthetic_retinal_fundus(640, 640)
    return "data:image/jpeg;base64," + base64.b64encode(image_to_bytes(img)).decode()


def create_assessment(client, headers, patient_id):
    res = client.post(f"{API}/assessments", headers=headers, json={
        "patientId": patient_id, "laterality": "OD", "imageDataUrl": image_data_url(),
        "fileSizeBytes": 1000, "filename": "fundus.jpg"})
    assert res.status_code == 201, res.text
    return res.json()


# ---------------------------------------------------------------------------
# 1, 2: creation by the Super Administrator
# ---------------------------------------------------------------------------

def test_01_superadmin_can_create_a_physician_account(test_client, superadmin_headers):
    user_id, temporary, body = create_physician(test_client, superadmin_headers, "dr.new")
    assert body["user"]["role"] == "PHYSICIAN"
    assert body["user"]["createdById"]
    assert len(temporary) >= 10
    listed = test_client.get(f"{API}/admin/users", headers=superadmin_headers).json()
    assert any(u["id"] == user_id for u in listed)


def test_02_admin_created_physician_is_active_and_must_change_password(test_client, superadmin_headers):
    user_id, temporary, body = create_physician(test_client, superadmin_headers, "dr.temp")
    assert body["user"]["status"] == "ACTIVE"
    assert body["user"]["mustChangePassword"] is True
    first = sign_in(test_client, "dr.temp", temporary)
    assert first["user"]["mustChangePassword"] is True
    # until the temporary password is changed, the clinical workflow is closed
    worklist = test_client.get(f"{API}/assessments", headers=bearer(first["access_token"]))
    assert worklist.status_code == 403 and "PASSWORD_CHANGE_REQUIRED" in worklist.json()["detail"]
    # the change is the one thing the account may do
    change_password(test_client, first["access_token"], temporary, "Dr-Temp-New-Pass-2026")
    # the token issued before the change no longer works; a fresh sign-in does
    stale = test_client.get(f"{API}/auth/me", headers=bearer(first["access_token"]))
    assert stale.status_code == 401
    again = sign_in(test_client, "dr.temp", "Dr-Temp-New-Pass-2026")
    assert again["user"]["mustChangePassword"] is False
    assert test_client.get(f"{API}/assessments", headers=bearer(again["access_token"])).status_code == 200


# ---------------------------------------------------------------------------
# 3, 4: self-registration is pending and locked out
# ---------------------------------------------------------------------------

def test_03_self_registration_creates_a_pending_account(test_client, superadmin_headers):
    res = register(test_client, "applicant.a")
    assert res.status_code == 201, res.text
    assert res.json()["status"] == "PENDING_APPROVAL"
    pending = test_client.get(f"{API}/admin/users/pending", headers=superadmin_headers).json()
    assert [u["username"] for u in pending] == ["applicant.a"]
    assert pending[0]["role"] == "PHYSICIAN"


def test_03b_public_registration_cannot_choose_a_role_or_status(test_client, superadmin_headers):
    res = register(test_client, "applicant.sneaky", role="SUPERADMIN", status="ACTIVE", activateImmediately=True)
    assert res.status_code == 201, res.text
    listed = test_client.get(f"{API}/admin/users?q=applicant.sneaky", headers=superadmin_headers).json()
    assert listed[0]["role"] == "PHYSICIAN" and listed[0]["status"] == "PENDING_APPROVAL"


def test_04_pending_account_cannot_sign_in_or_reach_protected_endpoints(test_client):
    register(test_client, "applicant.b")
    res = test_client.post(f"{API}/auth/login", json={"username": "applicant.b", "password": "Self-Reg-Pass-2026"})
    assert res.status_code == 403
    assert "awaiting administrative approval" in res.json()["detail"]
    # with no token at all the clinical endpoints are closed as before
    assert test_client.get(f"{API}/assessments").status_code == 401


# ---------------------------------------------------------------------------
# 5, 6: approval and rejection
# ---------------------------------------------------------------------------

def test_05_approval_permits_sign_in_and_the_assessment_workflow(test_client, superadmin_headers):
    register(test_client, "applicant.c")
    pending = test_client.get(f"{API}/admin/users/pending", headers=superadmin_headers).json()[0]
    res = test_client.patch(f"{API}/admin/users/{pending['id']}/approve", headers=superadmin_headers)
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "ACTIVE" and res.json()["approvedById"] and res.json()["approvedAt"]
    session = sign_in(test_client, "applicant.c", "Self-Reg-Pass-2026")
    assert session["user"]["mustChangePassword"] is False      # a self-chosen password is not temporary
    record = create_assessment(test_client, bearer(session["access_token"]), "PT-APPROVED-1")
    assert record["status"] in ("needs_review", "completed", "rejected")


def test_06_rejection_prevents_access(test_client, superadmin_headers):
    register(test_client, "applicant.d")
    pending = test_client.get(f"{API}/admin/users/pending", headers=superadmin_headers).json()[0]
    res = test_client.patch(f"{API}/admin/users/{pending['id']}/reject", headers=superadmin_headers,
                            json={"reason": "registration could not be confirmed"})
    assert res.status_code == 200 and res.json()["status"] == "REJECTED"
    login = test_client.post(f"{API}/auth/login", json={"username": "applicant.d", "password": "Self-Reg-Pass-2026"})
    assert login.status_code == 403 and "not approved" in login.json()["detail"]


# ---------------------------------------------------------------------------
# 7, 8: suspension and reactivation
# ---------------------------------------------------------------------------

def test_07_suspension_stops_an_existing_physician(test_client, superadmin_headers, physician_account):
    headers = physician_account["headers"]
    assert test_client.get(f"{API}/assessments", headers=headers).status_code == 200
    res = test_client.patch(f"{API}/admin/users/{physician_account['id']}/suspend", headers=superadmin_headers,
                            json={"reason": "leave of absence"})
    assert res.status_code == 200 and res.json()["status"] == "SUSPENDED"
    # the session issued before the suspension is dead, and a new sign-in is refused
    assert test_client.get(f"{API}/assessments", headers=headers).status_code == 401
    login = test_client.post(f"{API}/auth/login", json={
        "username": physician_account["username"], "password": physician_account["password"]})
    assert login.status_code == 403 and "suspended" in login.json()["detail"]


def test_08_reactivation_restores_access(test_client, superadmin_headers, physician_account):
    test_client.patch(f"{API}/admin/users/{physician_account['id']}/suspend", headers=superadmin_headers)
    res = test_client.patch(f"{API}/admin/users/{physician_account['id']}/reactivate", headers=superadmin_headers)
    assert res.status_code == 200 and res.json()["status"] == "ACTIVE"
    session = sign_in(test_client, physician_account["username"], physician_account["password"])
    assert test_client.get(f"{API}/assessments", headers=bearer(session["access_token"])).status_code == 200


# ---------------------------------------------------------------------------
# 9, 10: a physician's reach
# ---------------------------------------------------------------------------

def test_09_physician_cannot_use_superadmin_endpoints(test_client, physician_account):
    headers = physician_account["headers"]
    for method, path, body in (
        ("get", "/admin/users", None), ("get", "/admin/users/pending", None), ("get", "/admin/summary", None),
        ("get", "/admin/audit-events", None),
        ("post", "/admin/users", {"fullName": "X Y", "username": "x.y", "email": "x.y@example.org"}),
        ("post", "/admin/users/batch", {"count": 1, "usernamePrefix": "demo"}),
        ("patch", f"/admin/users/{physician_account['id']}/approve", None),
        ("patch", f"/admin/users/{physician_account['id']}/suspend", None),
        ("patch", f"/admin/users/{physician_account['id']}/reactivate", None),
        ("post", f"/admin/users/{physician_account['id']}/reset-password", None),
    ):
        kwargs = {"headers": headers}
        if body is not None:
            kwargs["json"] = body
        res = getattr(test_client, method)(f"{API}{path}", **kwargs)
        assert res.status_code == 403, f"{method.upper()} {path} answered {res.status_code} to a physician"


def test_10_physician_cannot_change_their_role_through_a_request(test_client, superadmin_headers, physician_account):
    headers = physician_account["headers"]
    # no route accepts a role: the profile and password routes ignore the field,
    # and the administration routes are closed to a physician
    me = test_client.get(f"{API}/auth/me", headers=headers).json()
    assert me["role"] == "PHYSICIAN"
    res = test_client.post(f"{API}/auth/change-password", headers=headers, json={
        "currentPassword": physician_account["password"], "newPassword": "Phys-One-Pass-2027",
        "newPasswordConfirmation": "Phys-One-Pass-2027", "role": "SUPERADMIN", "status": "ACTIVE"})
    assert res.status_code == 200 and res.json()["role"] == "PHYSICIAN"
    # the password change invalidated the earlier session (by design); sign in again
    headers = bearer(sign_in(test_client, physician_account["username"], "Phys-One-Pass-2027")["access_token"])
    for path in (f"/admin/users/{physician_account['id']}/approve", f"/admin/users/{physician_account['id']}/reactivate"):
        assert test_client.patch(f"{API}{path}", headers=headers, json={"role": "SUPERADMIN"}).status_code == 403
    listed = test_client.get(f"{API}/admin/users?q=physician.one", headers=superadmin_headers).json()
    assert listed[0]["role"] == "PHYSICIAN"


def test_10b_superadmin_is_not_granted_clinical_content(test_client, superadmin_headers):
    res = test_client.get(f"{API}/assessments", headers=superadmin_headers)
    assert res.status_code == 403 and "physician accounts only" in res.json()["detail"]


def test_10c_superadmin_cannot_approve_itself_and_registration_never_creates_a_superadmin(test_client, superadmin_account):
    res = test_client.patch(f"{API}/admin/users/{superadmin_account['id']}/approve", headers=superadmin_account["headers"])
    assert res.status_code in (403, 409)
    register(test_client, "wannabe.admin", role="SUPERADMIN")
    admins = test_client.get(f"{API}/admin/users?role=SUPERADMIN", headers=superadmin_account["headers"]).json()
    assert [u["username"] for u in admins] == [SUPERADMIN_CREDENTIALS["username"]]


# ---------------------------------------------------------------------------
# 11, 12: record separation
# ---------------------------------------------------------------------------

def test_11_physician_cannot_view_another_physicians_records(test_client, physician_account, second_physician_account):
    record = create_assessment(test_client, physician_account["headers"], "PT-PRIVATE-1")
    other = second_physician_account["headers"]
    for path in (f"/assessments/{record['id']}", f"/assessments/{record['id']}/status",
                 f"/assessments/{record['id']}/validation", f"/assessments/{record['id']}/result",
                 f"/assessments/{record['id']}/audit", f"/assessments/{record['id']}/report"):
        res = test_client.get(f"{API}{path}", headers=other)
        assert res.status_code == 404, f"{path} answered {res.status_code} to another physician"
    review = test_client.post(f"{API}/assessments/{record['id']}/review", headers=other, json={
        "agreement": "agree", "reviewerAssessedGrade": 2, "reviewerAssessedGradeLabel": "Grade 2"})
    assert review.status_code == 404
    assert all(r["id"] != record["id"] for r in test_client.get(f"{API}/assessments", headers=other).json())
    search = test_client.get(f"{API}/assessments/search?searchQuery=PT-PRIVATE", headers=other).json()
    assert search == []
    # the owner still sees it
    assert test_client.get(f"{API}/assessments/{record['id']}", headers=physician_account["headers"]).status_code == 200


def test_12_new_physician_starts_with_an_empty_history(test_client, superadmin_headers, physician_account):
    create_assessment(test_client, physician_account["headers"], "PT-EXISTING-1")
    fresh = activated_physician(test_client, superadmin_headers, "physician.fresh", "Phys-Fresh-Pass-2026")
    assert test_client.get(f"{API}/assessments", headers=fresh["headers"]).json() == []
    assert test_client.get(f"{API}/assessments/search", headers=fresh["headers"]).json() == []
    listed = {u["username"]: u for u in test_client.get(f"{API}/admin/users", headers=superadmin_headers).json()}
    assert listed["physician.fresh"]["assessmentCount"] == 0
    assert listed["physician.one"]["assessmentCount"] == 1


# ---------------------------------------------------------------------------
# 13: no secrets in any response
# ---------------------------------------------------------------------------

def test_13_no_password_hash_or_secret_is_returned(test_client, superadmin_headers, physician_account):
    forbidden = ("hashed_password", "hashedPassword", "password", "token_version", "tokenVersion")
    for path in ("/admin/users", "/admin/users/pending", f"/admin/users/{physician_account['id']}", "/admin/audit-events"):
        body = test_client.get(f"{API}{path}", headers=superadmin_headers).text
        for key in forbidden:
            assert f'"{key}"' not in body, f"{path} exposes {key}"
    me = test_client.get(f"{API}/auth/me", headers=physician_account["headers"]).text
    for key in forbidden:
        assert f'"{key}"' not in me
    # a temporary password appears once, in the response that issued it, and never in a listing
    _, temporary, _ = create_physician(test_client, superadmin_headers, "dr.once")
    assert temporary not in test_client.get(f"{API}/admin/users", headers=superadmin_headers).text
    assert temporary not in test_client.get(f"{API}/admin/audit-events", headers=superadmin_headers).text
    # the audit log never records an attempted password
    test_client.post(f"{API}/auth/login", json={"username": "dr.once", "password": "Guess-That-Secret-99"})
    assert "Guess-That-Secret-99" not in test_client.get(f"{API}/admin/audit-events", headers=superadmin_headers).text


# ---------------------------------------------------------------------------
# 14: audit
# ---------------------------------------------------------------------------

def test_14_administrative_events_are_audited(test_client, superadmin_account, superadmin_headers):
    register(test_client, "applicant.audit")
    pending = test_client.get(f"{API}/admin/users/pending", headers=superadmin_headers).json()[0]
    test_client.patch(f"{API}/admin/users/{pending['id']}/approve", headers=superadmin_headers)
    created_id, temporary, _ = create_physician(test_client, superadmin_headers, "dr.audited")
    first = sign_in(test_client, "dr.audited", temporary)
    change_password(test_client, first["access_token"], temporary, "Dr-Audited-Pass-2026")
    test_client.patch(f"{API}/admin/users/{created_id}/suspend", headers=superadmin_headers)
    test_client.patch(f"{API}/admin/users/{created_id}/reactivate", headers=superadmin_headers)
    test_client.post(f"{API}/admin/users/{created_id}/reset-password", headers=superadmin_headers)
    test_client.post(f"{API}/auth/login", json={"username": "dr.audited", "password": "wrong"})
    register(test_client, "applicant.rejected")
    rejected = [u for u in test_client.get(f"{API}/admin/users/pending", headers=superadmin_headers).json()
                if u["username"] == "applicant.rejected"][0]
    test_client.patch(f"{API}/admin/users/{rejected['id']}/reject", headers=superadmin_headers, json={"reason": "no"})

    events = test_client.get(f"{API}/admin/audit-events?limit=500", headers=superadmin_headers).json()
    types = {e["eventType"] for e in events}
    for required in ("SUPERADMIN_CREATED", "PHYSICIAN_REGISTRATION_SUBMITTED", "PHYSICIAN_ACCOUNT_CREATED",
                     "ACCOUNT_APPROVED", "ACCOUNT_REJECTED", "ACCOUNT_SUSPENDED", "ACCOUNT_REACTIVATED",
                     "TEMPORARY_PASSWORD_ISSUED", "PASSWORD_CHANGED", "SIGN_IN_SUCCESS", "SIGN_IN_FAILED"):
        assert required in types, f"no audit event of type {required}"
    suspended = [e for e in events if e["eventType"] == "ACCOUNT_SUSPENDED"][0]
    assert suspended["actorUserId"] == superadmin_account["id"]
    assert suspended["targetUserId"] == created_id
    assert suspended["outcome"] == "success" and suspended["timestamp"]
    failed = [e for e in events if e["eventType"] == "SIGN_IN_FAILED"][0]
    assert failed["outcome"] == "failure"
    # events for one account only
    mine = test_client.get(f"{API}/admin/audit-events?userId={created_id}", headers=superadmin_headers).json()
    assert mine and all(e["targetUserId"] == created_id or e["actorUserId"] == created_id for e in mine)


def test_14b_professional_review_is_audited(test_client, superadmin_headers, physician_account):
    record = create_assessment(test_client, physician_account["headers"], "PT-REVIEW-AUDIT")
    if record["status"] != "needs_review":
        pytest.skip("the synthetic image did not reach the review stage: %s" % record["status"])
    res = test_client.post(f"{API}/assessments/{record['id']}/review", headers=physician_account["headers"], json={
        "agreement": "agree", "reviewerAssessedGrade": record["modelObservation"]["primaryClassGrade"],
        "reviewerAssessedGradeLabel": "confirmed"})
    assert res.status_code == 200, res.text
    events = test_client.get(f"{API}/admin/audit-events?eventType=PROFESSIONAL_REVIEW_CONFIRMED",
                             headers=superadmin_headers).json()
    assert events and events[0]["assessmentId"] == record["id"] and events[0]["actorUserId"] == physician_account["id"]


# ---------------------------------------------------------------------------
# 15: the last Super Administrator
# ---------------------------------------------------------------------------

def test_15_last_active_superadmin_cannot_be_suspended(test_client, superadmin_account):
    res = test_client.patch(f"{API}/admin/users/{superadmin_account['id']}/suspend", headers=superadmin_account["headers"])
    assert res.status_code == 409
    assert "last active Super Administrator" in res.json()["detail"]
    assert test_client.get(f"{API}/auth/me", headers=superadmin_account["headers"]).status_code == 200


# ---------------------------------------------------------------------------
# bootstrap, batch creation, token hygiene
# ---------------------------------------------------------------------------

def test_bootstrap_is_idempotent_and_audited(test_client, superadmin_account):
    import asyncio
    from app.services import account_service
    from tests.conftest import TestingSessionLocal

    async def again():
        async with TestingSessionLocal() as session:
            return await account_service.bootstrap_superadmin(
                session, username="second.admin", email="second.admin@example.org", full_name="Second",
                initial_password="Second-Admin-Pass-2026", source="test")

    user, outcome = asyncio.run(again())
    assert user is None and "already exists" in outcome
    admins = test_client.get(f"{API}/admin/users?role=SUPERADMIN", headers=superadmin_account["headers"]).json()
    assert len(admins) == 1
    created = test_client.get(f"{API}/admin/audit-events?eventType=SUPERADMIN_CREATED", headers=superadmin_account["headers"]).json()
    assert len(created) == 1 and created[0]["targetUserId"] == superadmin_account["id"]


def test_batch_creation_for_a_supervised_demonstration(test_client, superadmin_headers):
    res = test_client.post(f"{API}/admin/users/batch", headers=superadmin_headers,
                           json={"count": 3, "usernamePrefix": "demo.phys", "facility": "Defence Demonstration"})
    assert res.status_code == 201, res.text
    created = res.json()["created"]
    assert [c["user"]["username"] for c in created] == ["demo.phys-01", "demo.phys-02", "demo.phys-03"]
    assert len({c["temporaryPassword"] for c in created}) == 3
    assert all(c["user"]["mustChangePassword"] and c["user"]["status"] == "ACTIVE" for c in created)
    # a second batch continues the numbering instead of failing on the taken names
    more = test_client.post(f"{API}/admin/users/batch", headers=superadmin_headers,
                            json={"count": 1, "usernamePrefix": "demo.phys"}).json()["created"]
    assert more[0]["user"]["username"] == "demo.phys-04"
    too_many = test_client.post(f"{API}/admin/users/batch", headers=superadmin_headers,
                                json={"count": 11, "usernamePrefix": "demo.phys"})
    assert too_many.status_code == 422


def test_token_carries_only_subject_role_and_version(test_client, physician_account):
    import jwt
    claims = jwt.decode(physician_account["token"], options={"verify_signature": False})
    assert set(claims) == {"sub", "role", "tv", "iat", "exp"}
    assert claims["role"] == "PHYSICIAN"


def test_sign_out_invalidates_the_token(test_client, physician_account):
    headers = physician_account["headers"]
    assert test_client.post(f"{API}/auth/logout", headers=headers).status_code == 204
    assert test_client.get(f"{API}/auth/me", headers=headers).status_code == 401


def test_expired_and_tampered_tokens_are_rejected(test_client, physician_account):
    import datetime
    from app.core.security import create_access_token
    expired = create_access_token(subject=physician_account["id"], expires_delta=datetime.timedelta(seconds=-5),
                                  extra_claims={"role": "PHYSICIAN", "tv": 0})
    assert test_client.get(f"{API}/auth/me", headers=bearer(expired)).status_code == 401
    head, payload, sig = physician_account["token"].split(".")
    assert test_client.get(f"{API}/auth/me", headers=bearer(f"{head}.{payload}x.{sig}")).status_code == 401


def test_registration_rejects_duplicates_and_weak_passwords(test_client):
    assert register(test_client, "dup.user").status_code == 201
    assert register(test_client, "dup.user").status_code == 409
    assert register(test_client, "other.user", email="dup.user@example.org").status_code == 409
    weak = register(test_client, "weak.user", password="onlyletters")
    assert weak.status_code == 422
    mismatch = register(test_client, "mm.user", passwordConfirmation="Different-Pass-2026")
    assert mismatch.status_code == 422
