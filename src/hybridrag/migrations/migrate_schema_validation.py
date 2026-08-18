"""
MongoDB Schema Validation Migration Script.

Applies JSON Schema validation to HybridRAG collections per MongoDB best practices (Rule 2.4).
This ensures data integrity at the database level.

Usage:
    python scripts/migrate_schema_validation.py

References:
    - https://mongodb.com/docs/manual/core/schema-validation/
    - MongoDB Schema Design Best Practices Rule 2.4
"""

import argparse
import os
import sys
from typing import Literal, TypedDict

# Load environment variables
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import OperationFailure

from hybridrag.engine.kg.mongo_impl import get_collection_name
from hybridrag.engine.namespace import NameSpace

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "hybridrag")


class SchemaValidationError(RuntimeError):
    """Schema validation could not be applied as requested."""


class ValidationResult(TypedDict):
    collection: str
    operation: Literal["create", "update"]
    validation_level: str
    validation_action: Literal["error", "errorAndLog"]


class ValidationDiagnostics(TypedDict):
    collection: str
    desired: dict
    effective: dict | None
    matches: bool


# Schema validators following MongoDB best practices
VALIDATORS = {
    # Conversation sessions - bounded document (Rule 1.1 compliant)
    # [Rule: validation-json-schema] additionalProperties must include ALL fields app writes
    "conversation_sessions": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["session_id", "created_at"],
            "properties": {
                "_id": {"description": "MongoDB document ID"},
                "session_id": {
                    "bsonType": "string",
                    "description": "Unique session identifier",
                },
                "message_count": {
                    "bsonType": "int",
                    "minimum": 0,
                    "description": "Count of messages (denormalized for quick access)",
                },
                "created_at": {
                    "bsonType": "date",
                    "description": "Session creation timestamp",
                },
                "updated_at": {
                    "bsonType": "date",
                    "description": "Last update timestamp",
                },
                "metadata": {"bsonType": "object", "description": "Session metadata"},
                "summary": {
                    "bsonType": "string",
                    "description": "Compacted conversation summary",
                },
                "summary_token_count": {
                    "bsonType": "int",
                    "minimum": 0,
                    "description": "Token count of summary",
                },
                "summary_updated_at": {
                    "bsonType": "date",
                    "description": "When summary was last updated",
                },
            },
            # M38: additionalProperties removed to allow flexible metadata
            # [Rule: mongodb-schema-design] Allow extensibility for future fields
        }
    },
    # Conversation messages - separate collection (Rule 1.1 fix)
    # [Rule: validation-json-schema] Include all fields app writes
    "conversation_messages": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["session_id", "role", "content", "timestamp"],
            "properties": {
                "_id": {"description": "MongoDB document ID"},
                "session_id": {
                    "bsonType": "string",
                    "description": "Reference to parent session",
                },
                "role": {
                    "enum": ["user", "assistant", "system"],
                    "description": "Message role",
                },
                "content": {"bsonType": "string", "description": "Message content"},
                "timestamp": {"bsonType": "date", "description": "Message timestamp"},
                "message_index": {
                    "bsonType": "int",
                    "minimum": 0,
                    "description": "Order within session",
                },
                "metadata": {
                    "bsonType": "object",
                    "description": "Optional message metadata",
                },
            },
        }
    },
    # Ingested documents - parent documents
    "ingested_documents": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["title", "source", "content", "created_at"],
            "properties": {
                "title": {
                    "bsonType": "string",
                    "minLength": 1,
                    "description": "Document title",
                },
                "source": {"bsonType": "string", "description": "Source URL or path"},
                "content": {
                    "bsonType": "string",
                    "description": "Full document content",
                },
                "format_type": {
                    "bsonType": "string",
                    "description": "Document format (markdown, html, etc.)",
                },
                "metadata": {
                    "bsonType": "object",
                    "description": "Embedded metadata (Rule 2.2 - data accessed together)",
                },
                "created_at": {"bsonType": "date"},
            },
        }
    },
    # Ingested chunks - with embeddings (Rule 3.2 - Extended Reference)
    "ingested_chunks": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["document_id", "content", "embedding", "chunk_index"],
            "properties": {
                "document_id": {
                    "bsonType": "objectId",
                    "description": "Reference to parent document",
                },
                "content": {"bsonType": "string", "description": "Chunk text content"},
                "embedding": {
                    "bsonType": "array",
                    "minItems": 1,
                    "description": "Vector embedding (1024 dimensions for Voyage)",
                },
                "chunk_index": {
                    "bsonType": "int",
                    "minimum": 0,
                    "description": "Position in document",
                },
                "token_count": {"bsonType": "int", "minimum": 0},
                "metadata": {
                    "bsonType": "object",
                    "description": "Cached parent fields (Rule 3.2 - Extended Reference)",
                    "properties": {
                        "title": {"bsonType": "string"},
                        "source": {"bsonType": "string"},
                        "chunk_method": {"bsonType": "string"},
                    },
                },
                "created_at": {"bsonType": "date"},
            },
        }
    },
}


RUNTIME_VALIDATORS = {
    NameSpace.KV_STORE_FULL_DOCS: {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "content"],
            "properties": {
                "_id": {"bsonType": "string"},
                "content": {"bsonType": "string"},
                "file_path": {"bsonType": "string"},
                "metadata": {"bsonType": "object"},
                "create_time": {"bsonType": ["int", "long", "double"]},
                "update_time": {"bsonType": ["int", "long", "double"]},
            },
        }
    },
    NameSpace.KV_STORE_TEXT_CHUNKS: {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "content", "full_doc_id"],
            "properties": {
                "_id": {"bsonType": "string"},
                "content": {"bsonType": "string"},
                "full_doc_id": {"bsonType": "string"},
                "file_path": {"bsonType": "string"},
                "metadata": {"bsonType": "object"},
                "llm_cache_list": {"bsonType": "array"},
            },
        }
    },
    NameSpace.VECTOR_STORE_CHUNKS: {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "content", "full_doc_id"],
            "properties": {
                "_id": {"bsonType": "string"},
                "content": {"bsonType": "string"},
                "full_doc_id": {"bsonType": "string"},
                "file_path": {"bsonType": "string"},
                "metadata": {"bsonType": "object"},
                "vector": {"bsonType": ["array", "binData"]},
                "created_at": {"bsonType": ["int", "long", "double"]},
            },
        }
    },
    NameSpace.DOC_STATUS: {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "status"],
            "properties": {
                "_id": {"bsonType": "string"},
                "status": {
                    "enum": [
                        "pending",
                        "processing",
                        "preprocessed",
                        "processed",
                        "failed",
                    ]
                },
                "file_path": {"bsonType": "string"},
                "metadata": {"bsonType": "object"},
                "chunks_list": {"bsonType": "array"},
            },
        }
    },
}


def get_validation_targets(workspace: str | None = None) -> dict[str, dict]:
    """Return standalone and actual workspaced runtime validation targets."""
    targets = dict(VALIDATORS)
    targets.update(
        {
            get_collection_name(workspace, namespace): validator
            for namespace, validator in RUNTIME_VALIDATORS.items()
        }
    )
    return targets


def apply_validation(
    db: Database,
    collection_name: str,
    validator: dict,
    validation_level: str = "moderate",
    validation_action: Literal["error", "errorAndLog"] = "error",
) -> ValidationResult:
    """
    Apply schema validation to an existing collection.

    Args:
        db: MongoDB database
        collection_name: Name of collection
        validator: JSON Schema validator
        validation_level: "strict" or "moderate"
            - strict: All inserts/updates must pass
            - moderate: Only new documents must pass (safer for migrations)
        validation_action: Reject invalid writes, optionally logging them first

    Returns:
        Applied collection policy and operation type.

    Raises:
        ValueError: If the requested action is invalid.
        SchemaValidationError: If MongoDB rejects the requested policy.
    """
    if validation_action not in {"error", "errorAndLog"}:
        raise ValueError("validation_action must be 'error' or 'errorAndLog'")
    try:
        db.command(
            {
                "collMod": collection_name,
                "validator": validator,
                "validationLevel": validation_level,
                "validationAction": validation_action,
            }
        )
        operation: Literal["create", "update"] = "update"
    except OperationFailure as e:
        if e.code != 26 and "ns not found" not in str(e).lower():
            raise SchemaValidationError(
                f"Failed to update schema validation for {collection_name}"
            ) from e
        try:
            db.create_collection(
                collection_name,
                validator=validator,
                validationLevel=validation_level,
                validationAction=validation_action,
            )
        except OperationFailure as create_error:
            raise SchemaValidationError(
                f"Failed to create {collection_name} with schema validation"
            ) from create_error
        operation = "create"

    return {
        "collection": collection_name,
        "operation": operation,
        "validation_level": validation_level,
        "validation_action": validation_action,
    }


def get_validation_diagnostics(
    db: Database,
    collection_name: str,
    validator: dict,
    validation_level: str,
    validation_action: Literal["error", "errorAndLog"],
) -> ValidationDiagnostics:
    """Compare requested schema-validation policy with collection options."""
    desired = {
        "validator": validator,
        "validationLevel": validation_level,
        "validationAction": validation_action,
    }
    try:
        info = db.command({"listCollections": 1, "filter": {"name": collection_name}})
    except OperationFailure as error:
        raise SchemaValidationError(
            f"Failed to inspect schema validation for {collection_name}"
        ) from error
    batch = info.get("cursor", {}).get("firstBatch", [])
    effective = None
    if batch:
        options = batch[0].get("options", {})
        effective = {
            "validator": options.get("validator"),
            "validationLevel": options.get("validationLevel"),
            "validationAction": options.get("validationAction"),
        }
    return {
        "collection": collection_name,
        "desired": desired,
        "effective": effective,
        "matches": effective == desired,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse schema-validation operator options.

    Args:
        argv: Optional argument list; process arguments are used when omitted.

    Returns:
        Parsed validation level and action.
    """
    parser = argparse.ArgumentParser(description="Apply HybridRAG schema validation")
    parser.add_argument(
        "--validation-action",
        choices=("error", "errorAndLog"),
        default=os.getenv("SCHEMA_VALIDATION_ACTION", "error"),
    )
    parser.add_argument(
        "--validation-level",
        choices=("moderate", "strict"),
        default=os.getenv("SCHEMA_VALIDATION_LEVEL", "moderate"),
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("MONGODB_WORKSPACE", ""),
        help="Workspace prefix used by runtime MongoDB collections",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Apply and verify validation policy for configured collections.

    Args:
        argv: Optional command-line argument list.

    Raises:
        SchemaValidationError: If a policy cannot be applied or verified.
    """
    args = parse_args(argv)
    if not MONGODB_URI:
        print("ERROR: MONGODB_URI environment variable not set")
        sys.exit(1)

    print("Connecting to MongoDB...")
    client = MongoClient(MONGODB_URI)
    try:
        db = client[DATABASE_NAME]

        print(f"\nApplying schema validation to database: {DATABASE_NAME}")
        print("=" * 60)

        targets = get_validation_targets(args.workspace)
        success_count = 0
        for collection_name, validator in targets.items():
            result = apply_validation(
                db,
                collection_name,
                validator,
                args.validation_level,
                args.validation_action,
            )
            diagnostics = get_validation_diagnostics(
                db,
                collection_name,
                validator,
                args.validation_level,
                args.validation_action,
            )
            if not diagnostics["matches"]:
                raise SchemaValidationError(
                    f"Effective validation policy differs for {collection_name}"
                )
            print(
                f"  [OK] {collection_name}: {result['operation']} "
                f"({result['validation_level']}, {result['validation_action']})"
            )
            success_count += 1

        print("=" * 60)
        print(f"Completed: {success_count}/{len(targets)} collections validated")
    finally:
        client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
