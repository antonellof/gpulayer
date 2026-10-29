import Link from "next/link";

/** GPUlayer mark: a GPU die on top of a stack of layers. Same paths as docs/assets/logo-mark.svg. */
export function LogoMark({ className = "h-6 w-6" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" fill="none" className={`${className} text-emerald-600 dark:text-emerald-400`} aria-hidden>
      <path d="M3 20.5 16 27.5 29 20.5" stroke="currentColor" strokeOpacity=".4" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3 15.5 16 22.5 29 15.5" stroke="currentColor" strokeOpacity=".7" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
      <path fill="currentColor" fillRule="evenodd" d="M16 3.2 29 10.2 16 17.2 3 10.2Z M16 7.4 21.2 10.2 16 13 10.8 10.2Z" />
      <path fill="currentColor" fillOpacity=".55" d="M16 8.9 18.4 10.2 16 11.5 13.6 10.2Z" />
    </svg>
  );
}

export function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2 tracking-tight" aria-label="GPUlayer home">
      <LogoMark />
      <span className="text-[17px]">
        <span className="font-bold">GPU</span>
        <span className="font-medium text-muted-foreground">layer</span>
      </span>
    </Link>
  );
}
