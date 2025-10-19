# Direction Consistency Checker for Correlated Pairs

def check_direction_consistency(positions, new_symbol, new_action, correlated_pairs):
    """
    Check if new trade direction is consistent with existing correlated positions.
    
    Args:
        positions: List of open MT5 positions
        new_symbol: Symbol for new trade
        new_action: 'buy' or 'sell'
        correlated_pairs: List of symbols correlated with new_symbol
    
    Returns:
        (bool, str): (can_trade, reason)
    """
    if not positions or not correlated_pairs:
        return True, "OK"
    
    # Check each open position
    for pos in positions:
        if pos.symbol in correlated_pairs:
            # Get position direction
            pos_action = 'buy' if pos.type == 0 else 'sell'  # 0=BUY, 1=SELL in MT5
            
            # If directions are opposite, block the trade
            if pos_action != new_action:
                return False, (
                    f"Direction conflict: {pos.symbol} is {pos_action.upper()}, "
                    f"but {new_symbol} signal is {new_action.upper()}. "
                    f"Correlated pairs should move in same direction!"
                )
    
    return True, "OK"


def get_crypto_pairs():
    """Return list of all crypto pairs"""
    return ['BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD', 'XRPUSD']


def is_crypto(symbol):
    """Check if symbol is cryptocurrency"""
    crypto_keywords = ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'ADA', 'DOT', 'LINK', 'UNI']
    return any(keyword in symbol.upper() for keyword in crypto_keywords)


def check_crypto_direction_consistency(positions, new_symbol, new_action):
    """
    Special check for crypto: ALL crypto should move in same direction.
    
    This is stricter than correlation check because crypto correlation is 80-90%.
    """
    if not is_crypto(new_symbol):
        return True, "OK"  # Not crypto, skip this check
    
    crypto_pairs = get_crypto_pairs()
    
    # Check all open crypto positions
    for pos in positions:
        if is_crypto(pos.symbol):
            pos_action = 'buy' if pos.type == 0 else 'sell'
            
            if pos_action != new_action:
                return False, (
                    f"CRYPTO DIRECTION CONFLICT: {pos.symbol} is {pos_action.upper()}, "
                    f"but {new_symbol} signal is {new_action.upper()}. "
                    f"All crypto pairs are highly correlated (80-90%) and should trade in same direction!"
                )
    
    return True, "OK"

