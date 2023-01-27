def calculate_mortgage_payment_per_month(p: float, r: float = 0.06/12, n: int = 360) -> float:
    """
    Calcuate the mortgage payment per month
    
    Args:
        p: mortgage amount
        r: monthly interest rate (default 6% per year)
        n: number of payment (default 360)
        
    Returns:
        monthly payment
    """
    
    numerator = r * (1 + r) ** n
    denominator = (1 + r) ** n - 1
    res = p * numerator / denominator
    return res


def calculate_maximum_mortgage_amount(s: float, r: float = 0.06/12, n: int = 360) -> float:
    """
    Calculate the maximum mortgage allowance
    
    Args:
        s: salary/income
        r: monthly interest rate (default 6% per year)
        n: number of payment (default 360)
        
    Returns:
        the maximum mortgage allowance
    """
    
    numerator = (1 + r) ** n - 1
    denominator = r * (1 + r) ** n
    m = s * 0.4
    res = m * numerator / denominator
    return res
