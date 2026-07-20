# SQL for ML Interviews

The job description says "excellent proficiency in SQL for data extraction and manipulation." This is tested directly. ML candidates often fail SQL screens because they memorize syntax but cannot handle real-world feature engineering queries.

## Core SQL Operations

### Joins

The most important skill. Know the difference.

| Join | Returns |
|---|---|
| INNER JOIN | Rows where key matches in both tables |
| LEFT JOIN | All rows from left, matched rows from right (NULL if no match) |
| RIGHT JOIN | All rows from right, matched rows from left |
| FULL OUTER JOIN | All rows from both, with NULLs where no match |
| CROSS JOIN | Cartesian product (every combination) |
| ANTI JOIN (LEFT JOIN WHERE right.key IS NULL) | Rows in left that have no match in right |
| SEMI JOIN (WHERE EXISTS) | Rows in left that have at least one match in right |

**Interview trap**: confusing LEFT JOIN + filter (drops unmatched) vs LEFT JOIN + filter on left table (preserves unmatched).

```sql
-- Correct: keep all left rows, even if no match
SELECT a.user_id, b.purchase_count
FROM users a
LEFT JOIN purchases b ON a.user_id = b.user_id
WHERE a.signup_date >= '2024-01-01';

-- Wrong: filter on b forces inner-join-like behavior
SELECT a.user_id, b.purchase_count
FROM users a
LEFT JOIN purchases b ON a.user_id = b.user_id
WHERE b.purchase_count > 0;  -- this turns it into an inner join
```

### Aggregations

`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `COUNT(DISTINCT ...)`.

Combine with `GROUP BY` to get per-group statistics. Use `HAVING` to filter groups.

```sql
SELECT user_id, COUNT(*) AS purchase_count, SUM(amount) AS total_spend
FROM purchases
WHERE purchase_date BETWEEN '2024-01-01' AND '2024-01-31'
GROUP BY user_id
HAVING COUNT(*) > 5;
```

### Subqueries

Use subqueries for:
- Filtering: `WHERE x IN (SELECT ...)`
- Existence: `WHERE EXISTS (SELECT 1 ...)`
- Scalar values: `WHERE x > (SELECT AVG(x) ...)`

Prefer CTEs for readability.

### Common Table Expressions (CTEs)

Make complex queries readable.

```sql
WITH active_users AS (
    SELECT user_id
    FROM sessions
    WHERE session_date >= CURRENT_DATE - INTERVAL '30 days'
),
recent_purchases AS (
    SELECT user_id, SUM(amount) AS total_spend
    FROM purchases
    WHERE purchase_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY user_id
)
SELECT a.user_id, COALESCE(p.total_spend, 0) AS spend_30d
FROM active_users a
LEFT JOIN recent_purchases p ON a.user_id = p.user_id;
```

### Window Functions

The most powerful SQL feature. Computes a value per row, related to a "window" of other rows.

```sql
SELECT
    user_id,
    purchase_date,
    amount,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY purchase_date) AS purchase_num,
    SUM(amount) OVER (PARTITION BY user_id ORDER BY purchase_date) AS cumulative_spend,
    LAG(amount) OVER (PARTITION BY user_id ORDER BY purchase_date) AS prev_amount,
    LEAD(amount) OVER (PARTITION BY user_id ORDER BY purchase_date) AS next_amount,
    RANK() OVER (PARTITION BY user_id ORDER BY amount DESC) AS amount_rank,
    AVG(amount) OVER (PARTITION BY user_id ORDER BY purchase_date
                      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_avg_7
FROM purchases;
```

**Window function concepts**:
- `PARTITION BY`: divide rows into groups (like GROUP BY, but per-row).
- `ORDER BY`: order within each partition.
- `ROWS BETWEEN ...`: physical window (e.g., last 7 rows).
- `RANGE BETWEEN ...`: logical window (e.g., last 7 days).
- `LAG` / `LEAD`: look at previous / next row.
- `FIRST_VALUE` / `LAST_VALUE` / `NTH_VALUE`: access specific values.
- `NTILE(n)`: divide rows into n equal buckets.

### Ranking and Deduplication

```sql
-- Keep only the most recent record per user
SELECT *
FROM (
    SELECT
        user_id,
        event_data,
        event_date,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_date DESC) AS rn
    FROM user_events
) ranked
WHERE rn = 1;
```

## Time-Based Queries

Most ML features are time-based. Master time filters and windows.

### Date functions (PostgreSQL syntax; similar in other DBs)

```sql
CURRENT_DATE                          -- today
CURRENT_TIMESTAMP                     -- now
DATE_TRUNC('day', order_date)        -- truncate to day
order_date::date                      -- cast to date
order_date + INTERVAL '7 days'        -- add time
EXTRACT(EPOCH FROM (end - start))    -- seconds between
DATE_PART('year', order_date)         -- extract part
```

### Leakage-safe feature extraction

The most common ML SQL question: "create features from 30 days before cutoff date."

```sql
-- Predict churn for users as of cutoff_date '2024-03-01'
-- Label: did the user have a session in [cutoff, cutoff + 30 days]?
-- Features: aggregations over [cutoff - 30, cutoff)

WITH features AS (
    SELECT
        user_id,
        COUNT(*) AS sessions_30d,
        SUM(duration) AS total_duration_30d,
        AVG(duration) AS avg_duration_30d,
        MAX(event_date) AS last_session_date,
        CURRENT_DATE - MAX(event_date) AS days_since_last_session
    FROM sessions
    WHERE event_date < '2024-03-01'
      AND event_date >= '2024-03-01' - INTERVAL '30 days'
    GROUP BY user_id
),
labels AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_date >= '2024-03-01' AND event_date < '2024-03-01' + INTERVAL '30 days'
                 THEN 1 ELSE 0 END) AS retained
    FROM sessions
    GROUP BY user_id
)
SELECT
    f.user_id,
    f.sessions_30d,
    f.total_duration_30d,
    f.avg_duration_30d,
    f.days_since_last_session,
    COALESCE(l.retained, 0) AS label
FROM features f
LEFT JOIN labels l ON f.user_id = l.user_id;
```

Key points:
- Features use `event_date < cutoff` (strict).
- Labels use `event_date >= cutoff` (no overlap with features).
- Window for features: 30 days ending at cutoff.
- Window for labels: 30 days starting at cutoff.
- The gap (if any) prevents leakage.

## Common ML SQL Patterns

### User-level aggregations

```sql
SELECT
    user_id,
    COUNT(*) AS total_events,
    COUNT(DISTINCT DATE(event_date)) AS active_days,
    SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) AS clicks,
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases,
    SUM(amount) AS total_revenue,
    AVG(amount) AS avg_order_value,
    MAX(amount) AS max_order_value,
    MIN(event_date) AS first_event,
    MAX(event_date) AS last_event
FROM events
GROUP BY user_id;
```

### Cohort analysis

```sql
SELECT
    DATE_TRUNC('month', signup_date) AS cohort,
    DATE_TRUNC('month', event_date) AS activity_month,
    COUNT(DISTINCT user_id) AS active_users,
    COUNT(DISTINCT user_id) FILTER (WHERE event_type = 'purchase') AS purchasers
FROM users u
JOIN events e ON u.user_id = e.user_id
WHERE signup_date >= '2023-01-01'
GROUP BY 1, 2
ORDER BY 1, 2;
```

### Funnel analysis

```sql
WITH step1 AS (
    SELECT DISTINCT user_id FROM events
    WHERE event_type = 'view_product' AND event_date >= '2024-01-01'
),
step2 AS (
    SELECT DISTINCT user_id FROM events
    WHERE event_type = 'add_to_cart' AND event_date >= '2024-01-01'
),
step3 AS (
    SELECT DISTINCT user_id FROM events
    WHERE event_type = 'checkout' AND event_date >= '2024-01-01'
),
step4 AS (
    SELECT DISTINCT user_id FROM events
    WHERE event_type = 'purchase' AND event_date >= '2024-01-01'
)
SELECT
    (SELECT COUNT(*) FROM step1) AS viewers,
    (SELECT COUNT(*) FROM step2) AS cart_adders,
    (SELECT COUNT(*) FROM step3) AS checkouts,
    (SELECT COUNT(*) FROM step4) AS purchasers;
```

### Time-windowed features (rolling)

```sql
SELECT
    user_id,
    event_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY user_id
        ORDER BY event_date
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
    ) AS spend_last_7_days,
    COUNT(*) OVER (
        PARTITION BY user_id
        ORDER BY event_date
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
    ) AS events_last_7_days
FROM events;
```

### A/B test analysis

```sql
-- Compute per-user metrics
WITH user_metrics AS (
    SELECT
        user_id,
        variant,
        COUNT(*) AS sessions,
        SUM(CASE WHEN converted THEN 1 ELSE 0 END) AS conversions
    FROM ab_test_data
    GROUP BY user_id, variant
)
-- Aggregate by variant
SELECT
    variant,
    COUNT(*) AS n_users,
    SUM(conversions) AS total_conversions,
    AVG(conversions) AS avg_conversions_per_user,
    SUM(conversions)::float / COUNT(*) AS conversion_rate,
    -- Approximate 95% CI
    1.96 * SQRT(
        (AVG(conversions) * (1 - AVG(conversions))) / COUNT(*)
    ) AS ci_half_width
FROM user_metrics
GROUP BY variant
ORDER BY variant;
```

## SQL Performance

For large datasets, query performance matters.

### Indexes

- Primary keys are auto-indexed.
- Foreign keys and frequently filtered columns should be indexed.
- Composite indexes for multi-column filters (order matters).

### Query optimization

- Filter early (in WHERE, not HAVING).
- Use appropriate JOIN types.
- Avoid SELECT * when you only need a few columns.
- Use LIMIT to test queries before running on full data.
- Avoid correlated subqueries when possible; use JOINs or window functions.
- Use EXPLAIN / EXPLAIN ANALYZE to understand query plans.

### Common pitfalls

- **Implicit type conversions**: `'123' = 123` may not work in all DBs; cast explicitly.
- **NULL handling**: NULL is not equal to anything, even NULL. Use `IS NULL`.
- **String comparison**: case-sensitive by default; use `LOWER()` or `ILIKE` for case-insensitive.
- **Date math**: date arithmetic differs across DBs (PostgreSQL uses `INTERVAL`, MySQL uses `DATE_ADD`).
- **Division by zero**: `a / NULLIF(b, 0)` instead of `a / b`.
- **Duplicates**: many-to-many joins can produce duplicates; use DISTINCT or aggregate first.

## SQL Interview Question Patterns

### 1. "Compute the second-highest value per group"

```sql
SELECT user_id, second_highest
FROM (
    SELECT
        user_id,
        amount AS second_highest,
        DENSE_RANK() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rnk
    FROM purchases
) ranked
WHERE rnk = 2;
```

### 2. "Find users who made a purchase in every month of 2024"

```sql
SELECT user_id
FROM purchases
WHERE EXTRACT(YEAR FROM purchase_date) = 2024
GROUP BY user_id
HAVING COUNT(DISTINCT DATE_TRUNC('month', purchase_date)) = 12;
```

### 3. "Compute month-over-month growth in revenue"

```sql
WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS revenue
    FROM orders
    GROUP BY 1
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
    (revenue - LAG(revenue) OVER (ORDER BY month))::float
        / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) AS mom_growth
FROM monthly_revenue
ORDER BY month;
```

### 4. "Find the top 3 users by total spend in each country"

```sql
WITH user_country_spend AS (
    SELECT
        u.user_id,
        u.country,
        SUM(o.amount) AS total_spend
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.user_id, u.country
),
ranked AS (
    SELECT
        user_id, country, total_spend,
        ROW_NUMBER() OVER (PARTITION BY country ORDER BY total_spend DESC) AS rnk
    FROM user_country_spend
)
SELECT user_id, country, total_spend
FROM ranked
WHERE rnk <= 3
ORDER BY country, rnk;
```

### 5. "Compute retention: for each signup cohort, what % of users are still active 30 days later?"

```sql
WITH cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('month', signup_date) AS cohort_month
    FROM users
),
activity AS (
    SELECT
        c.user_id,
        c.cohort_month,
        MAX(CASE WHEN s.session_date >= c.cohort_month + INTERVAL '30 days'
                 THEN 1 ELSE 0 END) AS retained_30d
    FROM cohorts c
    LEFT JOIN sessions s ON c.user_id = s.user_id
    GROUP BY c.user_id, c.cohort_month
)
SELECT
    cohort_month,
    COUNT(*) AS cohort_size,
    SUM(retained_30d) AS retained,
    AVG(retained_30d) AS retention_rate
FROM activity
GROUP BY cohort_month
ORDER BY cohort_month;
```

## What Senior Candidates Know

Senior candidates can:

1. Write leakage-safe feature extraction queries with proper cutoff dates.
2. Use window functions fluently (LAG, LEAD, RANK, rolling windows).
3. Use CTEs to make complex queries readable.
4. Compute A/B test metrics with proper aggregation.
5. Discuss query performance and indexing.
6. Handle nulls and edge cases explicitly.
7. Translate business questions into SQL without ambiguity.
8. Recognize when a problem is better solved in pandas/PySpark.

The biggest signal in SQL interviews is **clarity under ambiguity**: clarifying whether "all users" means active users, signed-up users, or users in the test, and using the right cutoff dates for time-based questions.
