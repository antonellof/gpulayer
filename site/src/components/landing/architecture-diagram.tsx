/** Static architecture diagram (inline SVG, follows the theme via currentColor/CSS vars). */
export function ArchitectureDiagram() {
  const box = "fill-[hsl(var(--card))] stroke-[hsl(var(--border))]";
  const accent = "fill-[hsl(var(--primary)/0.08)] stroke-[hsl(var(--primary)/0.6)]";
  const label = "fill-[hsl(var(--foreground))] text-[13px] font-semibold";
  const sub = "fill-[hsl(var(--muted-foreground))] text-[11px]";
  const arrow = "stroke-[hsl(var(--muted-foreground))]";

  return (
    <svg
      viewBox="0 0 760 470"
      role="img"
      aria-labelledby="arch-title arch-desc"
      className="h-auto w-full"
      style={{ fontFamily: "var(--font-sans), system-ui, sans-serif" }}
    >
      <title id="arch-title">GpuLayer architecture</title>
      <desc id="arch-desc">
        The SDK and CLI talk to a control plane with a scheduler, worker registry, leases, provisioning and
        storage. Local, existing and provider workers poll it and run commands, functions or torchrun.
      </desc>
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" className="fill-[hsl(var(--muted-foreground))]" />
        </marker>
      </defs>

      {/* clients */}
      <rect x="250" y="10" width="260" height="56" rx="8" className={box} />
      <text x="380" y="34" textAnchor="middle" className={label}>CLI · Python SDK · PyTorch</text>
      <text x="380" y="53" textAnchor="middle" className={sub}>gpulayer submit · gl.remote · gpulayer.pytorch</text>
      <line x1="380" y1="66" x2="380" y2="96" className={arrow} strokeWidth="1.5" markerEnd="url(#arr)" />

      {/* control plane */}
      <rect x="40" y="100" width="680" height="150" rx="10" className={accent} />
      <text x="60" y="124" className={label}>Control plane</text>
      <text x="168" y="124" className={sub}>embedded in your process, or `gpulayer server` over HTTPS</text>
      {[
        ["Scheduler", "capability-aware"],
        ["Worker registry", "heartbeats · leases"],
        ["Coordination", "ranks · rendezvous"],
        ["Provisioning", "provider adapters"],
      ].map(([t, s], i) => (
        <g key={t}>
          <rect x={60 + i * 165} y="140" width="150" height="48" rx="6" className={box} />
          <text x={135 + i * 165} y="160" textAnchor="middle" className={label}>{t}</text>
          <text x={135 + i * 165} y="177" textAnchor="middle" className={sub}>{s}</text>
        </g>
      ))}
      <rect x="60" y="198" width="315" height="36" rx="6" className={box} />
      <text x="217" y="221" textAnchor="middle" className={sub}>Metadata (jobs, tasks, events, logs)</text>
      <rect x="385" y="198" width="315" height="36" rx="6" className={box} />
      <text x="542" y="221" textAnchor="middle" className={sub}>Content-addressed artifacts</text>

      {/* arrows to workers */}
      {[150, 380, 610].map((x) => (
        <line key={x} x1={x} y1="252" x2={x} y2="288" className={arrow} strokeWidth="1.5" markerEnd="url(#arr)" />
      ))}
      <text x="392" y="275" className={sub}>poll · heartbeat · logs · results</text>

      {/* workers */}
      {[
        ["Local worker", "this machine"],
        ["Existing worker", "your GPU boxes"],
        ["Provider worker", "provisioned (simulated today)"],
      ].map(([t, s], i) => (
        <g key={t}>
          <rect x={45 + i * 230} y="292" width="210" height="56" rx="8" className={box} />
          <text x={150 + i * 230} y="316" textAnchor="middle" className={label}>{t}</text>
          <text x={150 + i * 230} y="335" textAnchor="middle" className={sub}>{s}</text>
        </g>
      ))}
      {[150, 380, 610].map((x) => (
        <line key={`w${x}`} x1={x} y1="348" x2={x} y2="380" className={arrow} strokeWidth="1.5" markerEnd="url(#arr)" />
      ))}

      {/* execution */}
      <rect x="45" y="384" width="670" height="72" rx="8" className={accent} />
      <text x="380" y="410" textAnchor="middle" className={label}>Process group per task</text>
      <text x="380" y="432" textAnchor="middle" className={sub}>
        any command · Python function · torchrun group, on CPU, CUDA or Apple Metal
      </text>
      <text x="380" y="448" textAnchor="middle" className={sub}>
        outputs and checkpoints flow back as artifacts
      </text>
    </svg>
  );
}
