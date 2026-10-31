# Shell job

Any executable or language works as a job. This one runs a shell script with `awk` on a CSV:

```bash
gpulayer submit --source examples/shell_job \
    --input examples/shell_job/sales.csv:sales.csv \
    -- bash report.sh
gpulayer job result <job-id> --download ./out      # out/totals.csv
```
