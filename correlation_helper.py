# Correlation and Currency Exposure Helper Functions

def get_currency_from_symbol(symbol):
    """Extract currencies from symbol"""
    # Handle different symbol formats
    symbol = symbol.upper()
    
    # Crypto
    if 'BTC' in symbol:
        return ['BTC']
    if 'ETH' in symbol:
        return ['ETH']
    if 'SOL' in symbol:
        return ['SOL']
    if 'DOGE' in symbol:
        return ['DOGE']
    if 'XRP' in symbol:
        return ['XRP']
    
    # Commodities
    if 'XAU' in symbol or 'GOLD' in symbol:
        return ['GOLD']
    if 'XAG' in symbol or 'SILVER' in symbol:
        return ['SILVER']
    if 'XTI' in symbol or 'OIL' in symbol or 'CL' in symbol:
        return ['OIL']
    
    # Forex - extract first 6 characters
    if len(symbol) >= 6:
        base = symbol[:3]
        quote = symbol[3:6]
        return [base, quote]
    
    return []

def check_currency_exposure(positions, new_symbol, max_exposure=3):
    """Check if adding new position would exceed currency exposure limit"""
    # Get currencies in new symbol
    new_currencies = get_currency_from_symbol(new_symbol)
    
    # Count current exposure
    currency_count = {}
    for pos in positions:
        currencies = get_currency_from_symbol(pos.symbol)
        for curr in currencies:
            currency_count[curr] = currency_count.get(curr, 0) + 1
    
    # Check if any currency would exceed limit
    for curr in new_currencies:
        current = currency_count.get(curr, 0)
        if current >= max_exposure:
            return False, f"{curr} exposure limit ({max_exposure}) would be exceeded"
    
    return True, "OK"

def get_correlated_pairs(symbol):
    """Get list of highly correlated pairs"""
    correlations = {
        'EURUSD': ['GBPUSD', 'AUDUSD', 'NZDUSD', 'EURGBP'],
        'GBPUSD': ['EURUSD', 'AUDUSD', 'NZDUSD', 'EURGBP'],
        'AUDUSD': ['NZDUSD', 'EURUSD', 'GBPUSD'],
        'NZDUSD': ['AUDUSD', 'EURUSD', 'GBPUSD'],
        'USDJPY': ['USDCHF', 'USDCAD'],
        'USDCHF': ['USDJPY', 'USDCAD'],
        'USDCAD': ['USDJPY', 'USDCHF'],
        'EURGBP': ['EURUSD', 'GBPUSD'],
        'EURJPY': ['EURUSD', 'USDJPY'],
        'GBPJPY': ['GBPUSD', 'USDJPY'],
        'AUDJPY': ['AUDUSD', 'USDJPY'],
        'NZDJPY': ['NZDUSD', 'USDJPY'],
        'EURAUD': ['EURUSD', 'AUDUSD'],
        'EURNZD': ['EURUSD', 'NZDUSD'],
        'GBPAUD': ['GBPUSD', 'AUDUSD'],
        'GBPNZD': ['GBPUSD', 'NZDUSD'],
        'AUDNZD': ['AUDUSD', 'NZDUSD'],
        'XAUUSD': ['XAGUSD'],  # Gold and silver correlated
        'XAGUSD': ['XAUUSD'],
        # Crypto - ALL highly correlated!
        'BTCUSD': ['ETHUSD', 'SOLUSD', 'DOGEUSD', 'XRPUSD'],
        'ETHUSD': ['BTCUSD', 'SOLUSD', 'DOGEUSD', 'XRPUSD'],
        'SOLUSD': ['BTCUSD', 'ETHUSD', 'DOGEUSD', 'XRPUSD'],
        'DOGEUSD': ['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD'],
        'XRPUSD': ['BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD'],
    }
    
    return correlations.get(symbol, [])

def check_correlation_limit(positions, new_symbol, max_correlated=2):
    """Check if adding new position would exceed correlation limit"""
    correlated_pairs = get_correlated_pairs(new_symbol)
    
    # Count how many correlated pairs are already open
    correlated_count = 0
    correlated_symbols = []
    
    for pos in positions:
        if pos.symbol in correlated_pairs:
            correlated_count += 1
            correlated_symbols.append(pos.symbol)
    
    if correlated_count >= max_correlated:
        return False, f"Correlation limit ({max_correlated}) exceeded. Open: {', '.join(correlated_symbols)}"
    
    return True, "OK"

