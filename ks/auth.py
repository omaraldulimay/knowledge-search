"""Demo users and the department-based permission rule.

Rule: everybody may read the department "alle" plus the departments of their own account.
This function is the only place that decides which departments a user may see.
"""

from dataclasses import dataclass

DEPARTMENTS = ("alle", "hr", "it", "vertrieb", "logistik")


@dataclass(frozen=True)
class User:
    name: str
    login: str
    departments: tuple[str, ...]  # own departments, without the shared "alle"


USERS: dict[str, User] = {
    u.login: u
    for u in [
        User("Mira Lindqvist", "mira", ("hr",)),
        User("Ole Brandt", "ole", ("it",)),
        User("Katja Sommer", "katja", ("vertrieb",)),
        User("Bent Carstens", "bent", ("logistik",)),
        User("Fiete Naumann", "fiete", ("hr", "it", "vertrieb", "logistik")),  # admin
    ]
}


def get_user(login: str) -> User:
    """Look a user up by login (case-insensitive). Raises KeyError for unknown users."""
    return USERS[login.strip().lower()]


def allowed_departments(user: User) -> frozenset[str]:
    """Departments whose documents this user may read."""
    return frozenset({"alle", *user.departments})
