from semantic_kernel.functions import kernel_function

_RATES = {
    ("USD", "INR"): 96.29,
    ("USD", "EUR"): 0.92,
    ("USD", "GBP"): 0.81,
}

class CurrencyPlugin:
    @kernel_function(description="Get the exchange rate from one currency to another (supports USD, INR, EUR, GBP).")
    def get_rate(self, from_currency: str, to_currency: str) -> float:
        pair = (from_currency.upper(), to_currency.upper())
        if pair[0] == pair[1]:
            return 1.0
        if pair in _RATES:
            return _RATES[pair]
        reversed_pair = (pair[1], pair[0])
        if reversed_pair in _RATES:
            return round(1 / _RATES[reversed_pair], 4)
        return 1.0
    
    @kernel_function(description="Convert an amount from one currency to another (supports USD, INR, EUR, GBP).")
    def convert(self, amount: float, rate: float) -> float:
        return round(amount * rate, 2)