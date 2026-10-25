class HarnessError(Exception):
    """An actionable error whose message is safe to put in run artifacts."""


class PolicyError(HarnessError):
    pass


class SourceError(HarnessError):
    pass


class LLMError(HarnessError):
    pass
