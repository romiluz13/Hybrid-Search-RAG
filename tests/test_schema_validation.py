from __future__ import annotations

import os
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pymongo import MongoClient
from pymongo.errors import OperationFailure

from hybridrag.migrations.migrate_schema_validation import (
    SchemaValidationError,
    apply_validation,
    get_validation_diagnostics,
    get_validation_targets,
    parse_args,
)


def test_apply_validation_preserves_error_default() -> None:
    db = Mock()

    result = apply_validation(db, "chunks", {"$jsonSchema": {}})

    assert result["operation"] == "update"
    command = db.command.call_args.args[0]
    assert command["validationAction"] == "error"


def test_apply_validation_accepts_error_and_log() -> None:
    db = Mock()

    result = apply_validation(
        db,
        "chunks",
        {"$jsonSchema": {}},
        validation_action="errorAndLog",
    )

    assert result["validation_action"] == "errorAndLog"
    command = db.command.call_args.args[0]
    assert command["validationAction"] == "errorAndLog"
    assert db.command.call_count == 1


def test_apply_validation_creates_missing_collection_with_same_policy() -> None:
    db = Mock()
    db.command.side_effect = OperationFailure("ns not found", code=26)

    result = apply_validation(
        db,
        "chunks",
        {"$jsonSchema": {}},
        validation_action="errorAndLog",
    )

    assert result["operation"] == "create"
    db.create_collection.assert_called_once_with(
        "chunks",
        validator={"$jsonSchema": {}},
        validationLevel="moderate",
        validationAction="errorAndLog",
    )


def test_apply_validation_propagates_typed_operation_failure() -> None:
    db = Mock()
    db.command.side_effect = OperationFailure("not authorized", code=13)

    with pytest.raises(SchemaValidationError, match="chunks"):
        apply_validation(db, "chunks", {"$jsonSchema": {}})


def test_apply_validation_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="validation_action"):
        apply_validation(
            Mock(),
            "chunks",
            {"$jsonSchema": {}},
            validation_action=cast(Any, "warn"),
        )


def test_schema_validation_cli_selects_action_and_level() -> None:
    args = parse_args(
        ["--validation-action", "errorAndLog", "--validation-level", "strict"]
    )

    assert args.validation_action == "errorAndLog"
    assert args.validation_level == "strict"


def test_runtime_validation_targets_use_workspace_collection_names() -> None:
    targets = get_validation_targets("tenant_a")

    assert {
        "tenant_a_full_docs",
        "tenant_a_text_chunks",
        "tenant_a_chunks",
        "tenant_a_doc_status",
    }.issubset(targets)
    assert "tenant_a_ingested_documents" not in targets
    assert targets["tenant_a_chunks"]["$jsonSchema"]["required"] == [
        "_id",
        "content",
        "full_doc_id",
    ]


def test_validation_diagnostics_compare_desired_and_effective_policy() -> None:
    validator = {"$jsonSchema": {"bsonType": "object"}}
    db = Mock()
    db.command.return_value = {
        "cursor": {
            "firstBatch": [
                {
                    "name": "chunks",
                    "options": {
                        "validator": validator,
                        "validationLevel": "strict",
                        "validationAction": "errorAndLog",
                    },
                }
            ]
        }
    }

    diagnostics = get_validation_diagnostics(
        db,
        "chunks",
        validator,
        "strict",
        "errorAndLog",
    )

    assert diagnostics["matches"] is True
    assert diagnostics["effective"] == diagnostics["desired"]


@pytest.mark.integration
def test_live_runtime_validator_rejects_invalid_create_and_collmod_writes() -> None:
    uri = os.getenv("HYBRIDRAG_LIVE_MONGODB_URI")
    if not uri:
        pytest.skip("HYBRIDRAG_LIVE_MONGODB_URI is not configured")

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    database_name = f"hybridrag_schema_test_{uuid4().hex}"
    collection_name = "tenant_chunks"
    validator = get_validation_targets("tenant")[collection_name]
    try:
        db = client[database_name]
        created = apply_validation(
            db,
            collection_name,
            validator,
            validation_level="strict",
            validation_action="error",
        )
        assert created["operation"] == "create"
        with pytest.raises(OperationFailure):
            db[collection_name].insert_one({"_id": "invalid"})

        updated = apply_validation(
            db,
            collection_name,
            validator,
            validation_level="strict",
            validation_action="errorAndLog",
        )
        assert updated["operation"] == "update"
        with pytest.raises(OperationFailure):
            db[collection_name].insert_one({"_id": "still-invalid"})
    finally:
        client.drop_database(database_name)
        client.close()
