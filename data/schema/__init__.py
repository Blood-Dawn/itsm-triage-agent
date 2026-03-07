# Re-export commonly used names so other modules can do:
#   from data.schema import Ticket, Category, Priority
# instead of:
#   from data.schema.ticket import Ticket, Category, Priority
#
# This is called a "public API" — the __init__.py controls what names
# are easily accessible. Internal helpers stay hidden unless someone
# deliberately imports them from the full path.

from data.schema.ticket import Ticket, Category, Priority, PRIORITY_WEIGHTS

__all__ = ["Ticket", "Category", "Priority", "PRIORITY_WEIGHTS"]
