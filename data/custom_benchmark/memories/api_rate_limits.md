# API Rate Limiting

## Current Limits
- Public endpoints: 100 requests/minute per IP
- Authenticated endpoints: 1000 requests/minute per user
- Webhook delivery: 50 requests/second aggregate

## Known Issues
The rate limiter uses a sliding window counter in Redis. Under heavy load (>10K RPM), the counter can drift by up to 5%. This is acceptable for now but needs fixing before we onboard enterprise clients.

## Third-Party API Limits
- Stripe: 100 requests/second (we're at ~40 in peak)
- SendGrid: 600 requests/minute for transactional email
- Twilio: 1 request/second for SMS (bottleneck for OTP flows)
