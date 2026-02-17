"""
MongoDB Persistence Layer for User Contexts

Design Principle:
  - Wraps and mirrors UserContextManager (doesn't replace it)
  - JSON files continue to work exactly as before
  - MongoDB is additive: optional dual-write (JSON + Mongo)
  - Silent-fail: if MongoDB is unavailable, falls back to JSON
"""

import os
from datetime import datetime
from typing import Optional

try:
    from pymongo import MongoClient, DESCENDING
    from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError
    _PYMONGO_AVAILABLE = True
except ImportError:
    _PYMONGO_AVAILABLE = False


def _get_config():
    """Read MongoDB config from environment variables"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    uri  = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db   = os.getenv("MONGO_DB",  "career_navigator")
    coll = os.getenv("MONGO_COLL", "users")
    return uri, db, coll


def _setup_indexes(col) -> None:
    """Create indexes for common queries"""
    try:
        col.create_index("user_id", unique=True)
        col.create_index([("last_updated", DESCENDING)])
        col.create_index("profile.resume_uploaded")
        col.create_index("career_state.current_target_role")
        col.create_index("readiness_assessment.status")
        col.create_index([("progress.last_activity_at", DESCENDING)])
    except Exception as e:
        print(f"[MongoStore] Index creation warning: {e}")


class MongoStore:
    """
    MongoDB mirror of the JSON user_context files.
    All methods are silent-fail: if MongoDB is unreachable, they return None/False
    and normal JSON flow continues unaffected.
    """

    def __init__(self):
        """Initialize MongoDB connection (safe-fail)"""
        self._col = None
        if not _PYMONGO_AVAILABLE:
            return
        
        try:
            uri, db_name, coll_name = _get_config()
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            # Connection test
            client.admin.command("ping")
            db = client[db_name]
            self._col = db[coll_name]
            _setup_indexes(self._col)
            print(f"[MongoStore] ✓ Connected → {db_name}.{coll_name}")
        except Exception as e:
            print(f"[MongoStore] ⚠ MongoDB unavailable ({str(e)[:50]}...) — using JSON only")

    @property
    def available(self) -> bool:
        """Check if MongoDB is available"""
        return self._col is not None

    # ─────────────────────────────────────────────────
    # CORE OPERATIONS
    # ─────────────────────────────────────────────────

    def sync(self, user_id: str, context: dict) -> bool:
        """Write full context to MongoDB (mirrors JSON save)"""
        if not self.available:
            return False
        try:
            doc = dict(context)
            doc.pop("_id", None)
            doc["user_id"] = user_id
            doc["last_updated"] = datetime.utcnow().isoformat()
            self._col.update_one(
                {"user_id": user_id},
                {"$set": doc},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"[MongoStore] sync error for {user_id}: {str(e)[:50]}")
            return False

    def load(self, user_id: str) -> Optional[dict]:
        """Load user context from MongoDB"""
        if not self.available:
            return None
        try:
            return self._col.find_one({"user_id": user_id}, {"_id": 0})
        except Exception as e:
            print(f"[MongoStore] load error: {str(e)[:50]}")
            return None

    def save_context(self, user_id: str, context: dict) -> bool:
        """Alias for sync() - matches UserContextManager API"""
        return self.sync(user_id, context)

    def load_context(self, user_id: str) -> Optional[dict]:
        """Alias for load() - matches UserContextManager API"""
        return self.load(user_id)

    def delete(self, user_id: str) -> bool:
        """Delete user context from MongoDB"""
        if not self.available:
            return False
        try:
            return self._col.delete_one({"user_id": user_id}).deleted_count > 0
        except Exception:
            return False

    def exists(self, user_id: str) -> bool:
        """Check if user exists in MongoDB"""
        if not self.available:
            return False
        try:
            return self._col.count_documents({"user_id": user_id}, limit=1) == 1
        except Exception:
            return False

    # ─────────────────────────────────────────────────
    # SECTION UPDATES (surgical updates)
    # ─────────────────────────────────────────────────

    def update_section(self, user_id: str, section: str, data: dict) -> bool:
        """Update a single top-level section"""
        if not self.available:
            return False
        try:
            self._col.update_one(
                {"user_id": user_id},
                {"$set": {
                    section: data,
                    "last_updated": datetime.utcnow().isoformat()
                }},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"[MongoStore] update_section error: {str(e)[:50]}")
            return False

    # ─────────────────────────────────────────────────
    # QUERIES (analytics)
    # ─────────────────────────────────────────────────

    def all_user_ids(self) -> list:
        """Get all user IDs"""
        if not self.available:
            return []
        try:
            return [d["user_id"] for d in self._col.find({}, {"user_id": 1, "_id": 0})]
        except Exception:
            return []

    def users_by_role(self, role: str) -> list:
        """Find users targeting a specific role"""
        if not self.available:
            return []
        try:
            return list(self._col.find(
                {"career_state.current_target_role": role},
                {"user_id": 1, "career_state": 1, "_id": 0}
            ))
        except Exception:
            return []

    def users_with_resume(self) -> list:
        """Get users who uploaded resumes"""
        if not self.available:
            return []
        try:
            return list(self._col.find(
                {"profile.resume_uploaded": True},
                {"user_id": 1, "profile.name": 1, "profile.skills": 1, "_id": 0}
            ))
        except Exception:
            return []

    def high_risk_users(self) -> list:
        """Get users with low readiness"""
        if not self.available:
            return []
        try:
            return list(self._col.find(
                {"readiness_assessment.confidence_score": {"$lt": 40}},
                {"user_id": 1, "career_state.current_target_role": 1, "_id": 0}
            ))
        except Exception:
            return []

    # ─────────────────────────────────────────────────
    # MIGRATION (one-time JSON → Mongo)
    # ─────────────────────────────────────────────────

    def migrate_from_json(self, context_dir: str) -> dict:
        """One-time migration: read all JSON files and insert to MongoDB"""
        import glob
        import json
        
        results = {"success": 0, "failed": 0, "skipped": 0}

        for fp in glob.glob(os.path.join(context_dir, "*_context.json")):
            try:
                with open(fp) as f:
                    ctx = json.load(f)
                user_id = ctx.get("user_id")
                if not user_id:
                    results["skipped"] += 1
                    continue
                if self.sync(user_id, ctx):
                    results["success"] += 1
                else:
                    results["failed"] += 1
            except Exception as e:
                print(f"[MongoStore] migrate error {fp}: {str(e)[:50]}")
                results["failed"] += 1

        print(f"[MongoStore] Migration complete: {results}")
        return results
