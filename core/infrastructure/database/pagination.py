from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar
from sqlalchemy import func, select
from sqlalchemy.orm import Session

T = TypeVar("T")

@dataclass(frozen=True)
class Page:
    page: int = 1
    page_size: int = 20
    def __post_init__(self):
        if self.page < 1:
            raise ValueError("page must be >= 1")
        if self.page_size < 1 or self.page_size > 100:
            raise ValueError("page_size must be between 1 and 100")
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


def paginate(session: Session, model, page: Page, *criteria, order_by=None) -> dict:
    total = session.scalar(select(func.count()).select_from(model).where(*criteria)) or 0
    stmt = select(model).where(*criteria)
    if order_by is not None:
        stmt = stmt.order_by(*order_by)
    stmt = stmt.offset(page.offset).limit(page.page_size)
    items = list(session.scalars(stmt).all())
    total_pages = (total + page.page_size - 1) // page.page_size
    return {"items": items, "pagination": {"page": page.page, "page_size": page.page_size,
            "total": total, "total_pages": total_pages,
            "has_previous": page.page > 1, "has_next": page.page < total_pages}}
