def compute_confidence(verifications):
    """
    Compute confidence as % of supported claims.
    """
    if not verifications:
        return 0
    supported = sum(1 for v in verifications if v['supported'])
    return int(100 * supported / len(verifications))
