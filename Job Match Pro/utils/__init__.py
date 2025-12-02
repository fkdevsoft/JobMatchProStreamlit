"""Utils package for Job Match Pro."""

from .storage import (
    load_jd_store,
    save_jd_to_store,
    delete_jd_from_store,
    get_jd_count,
    get_jd_by_id,
    clear_jd_store,
)

__all__ = [
    "load_jd_store",
    "save_jd_to_store",
    "delete_jd_from_store",
    "get_jd_count",
    "get_jd_by_id",
    "clear_jd_store",
]
