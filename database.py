"""
Database module for Google Sheets operations.
Handles face encoding storage, retrieval, and synchronization.
"""

import base64
import os
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gspread
from google.oauth2.service_account import Credentials

#credentials = Credentials.from_service_account_file(
 #   'credentials.json',  
  #  scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
#)

#gc = gspread.authorize(credentials)

logger = logging.getLogger(__name__)

# Sheet column headers
HEADERS = ["name", "face_encoding", "date_registered", "last_seen", "face_id"]

# Local cache file for offline mode
CACHE_FILE = "face_cache.pkl"


class FaceDatabase:
    """Manages face data persistence using Google Sheets."""

    def __init__(
        self,
        credentials_path: str = "credentials.json",
        spreadsheet_name: str = "1E6QOsZQx3vRpl5P49mXCmdzOvJhlbYhAL7hyZnkaCGI",
    ):
        """Initialize the database connection."""
        self.credentials_path = credentials_path
        self.spreadsheet_name = spreadsheet_name
        self.gc: Optional[gspread.Client] = None
        self.sheet = None
        self.worksheet = None
        self._local_cache: List[Dict] = []
        self._connected = False

    def connect(self) -> bool:
        """Establish connection to Google Sheets."""
        try:
            if not os.path.exists(self.credentials_path):
                logger.warning(
                    f"Credentials file not found at {self.credentials_path}. "
                    "Operating in offline mode with local cache."
                )
                self._load_local_cache()
                return False

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                ]

            credentials = Credentials.from_service_account_file(
                self.credentials_path, scopes=scopes
            )
            self.gc = gspread.authorize(credentials)

            # Use open_by_key for spreadsheet ID
            spreadsheet = self.gc.open_by_key(self.spreadsheet_name)
            self.worksheet = spreadsheet.sheet1

            if not self.worksheet.get_all_records():
                self.worksheet.append_row(HEADERS)
                logger.info("Initialized database with headers")

            self._connected = True
            logger.info("Successfully connected to Google Sheets")
            return True

        except gspread.SpreadsheetNotFound:
            logger.error(
                f"Spreadsheet '{self.spreadsheet_name}' not found or no access. "
                "Make sure the sheet is shared with the service account."
            )
            self._load_local_cache()
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets: {e}")
            self._load_local_cache()
            return False

    def _load_local_cache(self) -> None:
        """Load face data from local cache file."""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "rb") as f:
                    self._local_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._local_cache)} faces from local cache")
            else:
                self._local_cache = []
        except Exception as e:
            logger.error(f"Failed to load local cache: {e}")
            self._local_cache = []

    def _save_local_cache(self) -> None:
        """Save face data to local cache file."""
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(self._local_cache, f)
        except Exception as e:
            logger.error(f"Failed to save local cache: {e}")

    @staticmethod
    def encode_face(encoding) -> str:
        """Serialize face encoding to base64 string for storage."""
        bytes_data = pickle.dumps(encoding)
        return base64.b64encode(bytes_data).decode("utf-8")

    @staticmethod
    def decode_face(encoded_str: str):
        """Deserialize face encoding from base64 string."""
        bytes_data = base64.b64decode(encoded_str.encode("utf-8"))
        return pickle.loads(bytes_data)

    def get_all_faces(self) -> List[Dict]:
        """Retrieve all known faces from database."""
        if self._connected and self.worksheet:
            try:
                records = self.worksheet.get_all_records()
                faces = []
                for idx, record in enumerate(records):
                    try:
                        if record.get("name") and record.get("face_encoding"):
                            face_data = {
                                "name": record["name"],
                                "face_encoding": self.decode_face(
                                    record["face_encoding"]
                                ),
                                "date_registered": record.get("date_registered", ""),
                                "last_seen": record.get("last_seen", ""),
                                "face_id": record.get("face_id", str(idx)),
                            }
                            faces.append(face_data)
                    except Exception as e:
                        logger.warning(f"Skipping invalid record: {e}")
                        continue
                self._local_cache = faces
                self._save_local_cache()
                return faces
            except Exception as e:
                logger.error(f"Failed to fetch from Google Sheets: {e}")
                return self._local_cache
        return self._local_cache

    def register_face(
        self, name: str, face_encoding, face_id: Optional[str] = None
    ) -> bool:
        """Register a new face in the database."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        encoded_encoding = self.encode_face(face_encoding)

        if not face_id:
            face_id = str(hash(name + now))[:8]

        face_data = {
            "name": name,
            "face_encoding": face_encoding,
            "date_registered": now,
            "last_seen": now,
            "face_id": face_id,
        }
        self._local_cache.append(face_data)
        self._save_local_cache()

        if self._connected and self.worksheet:
            try:
                self.worksheet.append_row(
                    [name, encoded_encoding, now, now, face_id]
                )
                logger.info(f"Registered face for '{name}' in Google Sheets")
                return True
            except Exception as e:
                logger.error(f"Failed to register in Google Sheets: {e}")

        logger.info(f"Registered face for '{name}' in local cache (offline mode)")
        return True

    def update_last_seen(self, name: str, face_id: str) -> bool:
        """Update the last_seen timestamp for a face."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for face in self._local_cache:
            if face["face_id"] == face_id or face["name"] == name:
                face["last_seen"] = now
                break
        self._save_local_cache()

        if self._connected and self.worksheet:
            try:
                records = self.worksheet.get_all_records()
                for idx, record in enumerate(records):
                    if record.get("face_id") == face_id or record.get("name") == name:
                        row_num = idx + 2
                        self.worksheet.update_cell(row_num, 4, now)
                        return True
            except Exception as e:
                logger.error(f"Failed to update last_seen in Google Sheets: {e}")

        return True

    def sync_offline_data(self) -> Tuple[int, int]:
        """Synchronize locally cached faces to Google Sheets."""
        if not self._connected or not self.worksheet:
            logger.warning("Cannot sync: Not connected to Google Sheets")
            return (0, 0)

        success = 0
        errors = 0

        try:
            existing_names = set()
            records = self.worksheet.get_all_records()
            for record in records:
                if record.get("name"):
                    existing_names.add(record["name"])

            for face in self._local_cache:
                if face["name"] not in existing_names:
                    try:
                        encoded = self.encode_face(face["face_encoding"])
                        self.worksheet.append_row(
                            [
                                face["name"],
                                encoded,
                                face["date_registered"],
                                face["last_seen"],
                                face["face_id"],
                            ]
                        )
                        success += 1
                    except Exception as e:
                        logger.error(f"Failed to sync face '{face['name']}': {e}")
                        errors += 1

            logger.info(f"Synced {success} faces, {errors} errors")
            return (success, errors)

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return (0, errors + 1)

    def close(self) -> None:
        """Close the database connection."""
        if self.gc:
            self.gc.session.close()
            logger.info("Database connection closed")


# Singleton instance
_db_instance: Optional[FaceDatabase] = None


def get_database(
    credentials_path: str = "credentials.json",
    spreadsheet_name: str = "1E6QOsZQx3vRpl5P49mXCmdzOvJhlbYhAL7hyZnkaCGI",
) -> FaceDatabase:
    """Get or create the database singleton instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = FaceDatabase(credentials_path, spreadsheet_name)
    return _db_instance