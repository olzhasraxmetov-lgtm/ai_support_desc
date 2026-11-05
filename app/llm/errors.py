class LLMClientError(RuntimeError):
    def __init__(self, message: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id

    def __str__(self) -> str:
        message = super().__str__()
        if self.request_id is None:
            return message
        return f"{message} ID запроса: {self.request_id}"

class LLMAccessError(LLMClientError):
    pass

class LLMBalanceError(LLMClientError):
    pass

class LLMRequestError(LLMClientError):
    pass

class LLMRateLimitError(LLMClientError):
    pass

class LLMTimeoutError(LLMClientError):
    pass

class LLMConnectionError(LLMClientError):
    pass

class LLMProviderError(LLMClientError):
    pass