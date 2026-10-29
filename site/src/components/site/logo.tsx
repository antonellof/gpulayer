import Link from "next/link";

export function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
      <svg viewBox="0 0 24 24" className="h-6 w-6 text-emerald-600 dark:text-emerald-400" aria-hidden>
        <rect x="2" y="14" width="20" height="6" rx="1.5" fill="currentColor" opacity="0.35" />
        <rect x="4" y="9" width="16" height="5" rx="1.5" fill="currentColor" opacity="0.6" />
        <rect x="6" y="4" width="12" height="5" rx="1.5" fill="currentColor" />
      </svg>
      <span>GpuLayer</span>
    </Link>
  );
}
