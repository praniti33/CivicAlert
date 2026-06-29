# Security Specification for CivicAlert Pune

## 1. Data Invariants
- An issue must have a unique ID.
- An issue must have a valid category: Pothole, Water Leakage, Streetlight, Garbage, or Other.
- An issue status must be either Reported, In Progress, or Resolved.
- A user can read all issues without authentication (to support open local collaboration).
- A user can create an issue.
- A user can update an issue (upvotes, status, department, estimated resolution date, comments).

## 2. The "Dirty Dozen" Payloads (Denial/Exploit Scenarios)
1. Creating an issue with an invalid category (e.g., "UFO Sightings").
2. Creating an issue with a status of "Resolved" on day one without resolution.
3. Updating the upvotes of an issue to a negative value or string.
4. Overwriting or deleting existing issues entirely (delete should be forbidden).
5. Injecting script tags or HTML into the description (prevented via validation or client-side rendering).
6. Modifying fields of an issue with a status of 'Resolved' (Terminal state locking).
7. Creating a document with a non-string or 1MB string ID.
8. Bypassing schema by adding shadow fields (e.g., `isVerifiedAdmin: true`).
9. Deleting an issue document.
10. Creating comments with missing required keys.
11. Setting coordinates `lat` and `lng` to values out of range.
12. Attempting to set or modify system-only attributes if any exist.

## 3. Test Cases Verified
- All public reads to `/issues` are allowed.
- Writes conforming to the schema are allowed.
- Deletions are blocked.
- Invalid categories or statuses are blocked.
