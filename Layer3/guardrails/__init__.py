from .input_guard import InputGuardrail, InputResult
from .output_guard import OutputGuardrail, OutputResult

input_guard  = InputGuardrail()
output_guard = OutputGuardrail()

__all__ = [
    "InputGuardrail", "InputResult",
    "OutputGuardrail", "OutputResult",
    "input_guard", "output_guard",
]
