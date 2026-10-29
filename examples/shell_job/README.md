# Shell job

Any executable or language works as a job. This one runs a shell script with `awk` on a CSV:

```bash
gpulayer submit --source examples/shell_job \
    --input examples/shell_job/sales.csv:sales.csv \
    -- bash report.sh
gpulayer job result <job-id> --download ./out      # out/totals.csv
```

The same form runs GPU binaries. For example, on a worker with an NVIDIA GPU and an
ffmpeg build with NVENC:

```bash
gpulayer submit --accelerator cuda --input clip.mov:clip.mov -- \
    bash -c 'ffmpeg -hwaccel cuda -i "$GPULAYER_INPUT_DIR/clip.mov" -c:v h264_nvenc "$GPULAYER_OUTPUT_DIR/clip.mp4"'
```

(The ffmpeg command is an illustration and has not been run in this repository.)
