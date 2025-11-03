from prometheus_client import Counter

# Example metric
requests_total = Counter(
    "homebot_requests_total", "Total number of processed requests"
)
