# Account and Device Security Policy

## Multi-factor authentication

Multi-factor authentication is required for email, source control, cloud consoles,
and the company VPN. Hardware security keys are preferred for administrators and
employees with access to production systems.

## Lost devices

A lost or stolen company device must be reported to the security team within one
hour of discovery. Employees should not attempt to remotely erase a device unless
the security team requests it, because evidence may be needed for investigation.

## Secrets

API keys and passwords must never be committed to source control or pasted into
tickets. Store development secrets in the approved password manager and rotate a
secret immediately if it may have been exposed.
