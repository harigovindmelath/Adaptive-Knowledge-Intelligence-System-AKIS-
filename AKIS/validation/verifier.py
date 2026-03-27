import logging

def verify_claims(claims, retrieved_chunks):
    """
    Verify each claim against retrieved context.
    Returns: [{claim, supported, source}]
    """
    results = []
    for claim in claims:
        supported = False
        source = None
        for chunk in retrieved_chunks:
            if claim.lower() in chunk['text'].lower():
                supported = True
                source = chunk['chunk_id']
                break
        results.append({'claim': claim, 'supported': supported, 'source': source})
    return results
