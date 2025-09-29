CONTACT_ADMIN = "Please contact admin of the api"


class HasuraException(Exception):
    def __init__(self, status_code: int, message: str = CONTACT_ADMIN):
        self.status_code = status_code
        self.message = message
