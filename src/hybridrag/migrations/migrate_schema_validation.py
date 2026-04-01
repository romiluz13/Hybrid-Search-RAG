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

import os
import sys

# Load environment variables
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import OperationFailure

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "hybridrag")


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


def apply_validation(
    db, collection_name: str, validator: dict, validation_level: str = "moderate"
):
    """
    Apply schema validation to an existing collection.

    Args:
        db: MongoDB database
        collection_name: Name of collection
        validator: JSON Schema validator
        validation_level: "strict" or "moderate"
            - strict: All inserts/updates must pass
            - moderate: Only new documents must pass (safer for migrations)
    """
    try:
        db.command(
            {
                "collMod": collection_name,
                "validator": validator,
                "validationLevel": validation_level,
                "validationAction": "error",
            }
        )
        print(f"  [OK] {collection_name}: Validation applied ({validation_level})")
        return True
    except OperationFailure as e:
        if "ns not found" in str(e):
            print(f"  [SKIP] {collection_name}: Collection does not exist")
        else:
            print(f"  [ERROR] {collection_name}: {e}")
        return False


def main():
    if not MONGODB_URI:
        print("ERROR: MONGODB_URI environment variable not set")
        sys.exit(1)

    print("Connecting to MongoDB...")
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]

    print(f"\nApplying schema validation to database: {DATABASE_NAME}")
    print("=" * 60)

    success_count = 0
    for collection_name, validator in VALIDATORS.items():
        # Use "moderate" for existing data compatibility
        if apply_validation(db, collection_name, validator, "moderate"):
            success_count += 1

    print("=" * 60)
    print(f"Completed: {success_count}/{len(VALIDATORS)} collections validated")

    # Verify by checking collection options
    print("\nVerifying validators...")
    for collection_name in VALIDATORS.keys():
        try:
            info = db.command(
                {"listCollections": 1, "filter": {"name": collection_name}}
            )
            if info.get("cursor", {}).get("firstBatch"):
                opts = info["cursor"]["firstBatch"][0].get("options", {})
                if "validator" in opts:
                    print(f"  [VERIFIED] {collection_name}")
                else:
                    print(f"  [NO VALIDATOR] {collection_name}")
        except Exception as e:
            print(f"  [ERROR] {collection_name}: {e}")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
