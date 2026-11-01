# Cost accounting

`CostRecord` keeps estimates and charges apart:

| `kind` | Meaning |
|---|---|
| `estimate` | from a provider `PriceQuote` at provisioning time |
| `provider_usage` / `provider_charge` | reserved for real adapters that report usage/billing |
| `internal` | GPUlayer-calculated accounting |

Every record has `amount`, `currency`, `unit` (`per_hour` or `total`), `source` and
`recorded_at`. A missing price is recorded as unknown. When `max_hourly_cost` is set, an offer
without a price is rejected.

Provider adapters attach a `PriceQuote` to an offer when the provider publishes a price. When
GPUlayer provisions an instance for a job, it records an `estimate`, shown in
`JobResult.costs` (`gpulayer job result <id>`). No real provider adapter exists yet, so only the
simulated `FakeProvider` produces estimates, labelled as simulated. Workers that joined with
`gpulayer join` have no price.
