```
# Storage structure
store = {
    key: {
        value: any,
        expires_at: timestamp | null
    }
}

# SET with TTL
function set(key, value, ttl_seconds=null):
    expires_at = null
    if ttl_seconds:
        expires_at = now() + ttl_seconds
    store[key] = {value: value, expires_at: expires_at}

# GET with lazy expiration check
function get(key):
    if key not in store:
        return null
    entry = store[key]
    if entry.expires_at and now() > entry.expires_at:
        delete store[key]
        return null
    return entry.value

# Optional: background cleanup (run periodically)
function cleanup_expired():
    for key in store.keys():
        if store[key].expires_at and now() > store[key].expires_at:
            delete store[key]
```

---

Two common strategies:
- **Lazy expiration** (check on read) - simple, no background work, but dead keys linger until accessed
- **Active expiration** (background sweep) - cleaner memory, but adds complexity/overhead

Most production systems do both - lazy on read + periodic background cleanup for keys nobody's asking for.