"""
Deefake - Social Media URL Filter

Parses a list of URLs from Vision API results and identifies
which ones belong to monitored social media platforms:
  - twitter.com / x.com
  - reddit.com
  - facebook.com
  - instagram.com

Also identifies "High-Impact Spreaders" — social platforms
where the content has been found.
"""

from urllib.parse import urlparse


# Platforms to monitor and their domain patterns
SOCIAL_PLATFORMS = {
    "twitter": ["twitter.com", "x.com", "t.co"],
    "reddit": ["reddit.com"],
    "facebook": ["facebook.com", "fb.com", "fb.me"],
    "instagram": ["instagram.com"],
}


def filter_social_urls(url_list: list) -> dict:
    """
    Parse a list of URLs and count matches per social platform.

    Args:
        url_list: list of URL strings

    Returns:
        {
            "social_urls": [
                {"url": str, "platform": str}
            ],
            "social_count": int,
            "by_platform": {
                "twitter": int,
                "reddit": int,
                "facebook": int,
                "instagram": int
            },
            "high_impact_spreaders": [str]  # platform names with > 0 matches
        }
    """
    by_platform = {name: 0 for name in SOCIAL_PLATFORMS}
    social_urls = []

    for url in url_list:
        platform = _classify_social(url)
        if platform:
            by_platform[platform] += 1
            social_urls.append({"url": url, "platform": platform})

    social_count = sum(by_platform.values())
    high_impact = [name for name, count in by_platform.items() if count > 0]

    return {
        "social_urls": social_urls,
        "social_count": social_count,
        "by_platform": by_platform,
        "high_impact_spreaders": high_impact,
    }


def _classify_social(url: str) -> str | None:
    """Classify a URL into a social media platform, or None."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        for platform, patterns in SOCIAL_PLATFORMS.items():
            for pattern in patterns:
                if domain == pattern or domain.endswith("." + pattern):
                    return platform
    except Exception:
        pass

    return None


def extract_unique_domains(url_list: list) -> list:
    """Extract unique base domains from a list of URLs."""
    domains = set()
    for url in url_list:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            if domain:
                domains.add(domain)
        except Exception:
            pass
    return sorted(domains)
