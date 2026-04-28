"""
Deefake - Reach Score Calculator

Implements the Spreadness Score formula:

    Reach Score = (Total Unique Domains) + (Social Media URL Count × 2)

Risk classification:
    > 50 → Critical
    > 30 → High
    > 15 → Medium
    else → Low
"""


def calculate_reach_score(unique_domains: int, social_count: int, detected_domains: list = None) -> dict:
    """
    Calculate an optimized Reach (Spreadness) Score with platform authority.
    """
    if detected_domains is None:
        detected_domains = []

    # Platform Authority Weights
    HIGH_AUTHORITY = ["nytimes.com", "bbc.com", "cnn.com", "reuters.com", "gov", "edu", "apnews.com"]
    SOCIAL_PLATFORMS = ["twitter.com", "x.com", "facebook.com", "instagram.com", "reddit.com", "youtube.com", "tiktok.com"]
    
    authority_score = 0
    for domain in detected_domains:
        domain_lower = domain.lower()
        if any(auth in domain_lower for auth in HIGH_AUTHORITY):
            authority_score += 5  # High impact news/gov
        elif any(social in domain_lower for social in SOCIAL_PLATFORMS):
            authority_score += 3  # Social viral impact
        else:
            authority_score += 1  # Standard web mention

    # Hybrid Formula: Base unique domains + Social weight + Authority weight
    base_score = unique_domains + (social_count * 2)
    final_score = base_score + authority_score

    if final_score > 75:
        risk_level = "Critical"
    elif final_score > 40:
        risk_level = "High"
    elif final_score > 20:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "score": final_score,
        "base_score": base_score,
        "authority_bonus": authority_score,
        "unique_domains": unique_domains,
        "social_count": social_count,
        "risk_level": risk_level,
        "formula": f"Base({base_score}) + Authority({authority_score}) = {final_score}",
    }
