"use client";

// Dashboard screenshots in one row; a click opens the image full size over the page
// (click, Escape or the close button to close).

import { useEffect, useState } from "react";
import { X, ZoomIn } from "lucide-react";

type Shot = { dark: string; light?: string; caption: string };

const SHOTS: Shot[] = [
  {
    light: "/docs-assets/dashboard/overview-light.png",
    dark: "/docs-assets/dashboard/overview-dark.png",
    caption: "Overview: running and finished jobs, workers and their GPUs",
  },
  { dark: "/docs-assets/dashboard/job-ddp-dark.png", caption: "A distributed job: its placement and every rank's logs" },
  { dark: "/docs-assets/dashboard/workers-dark.png", caption: "The workers: their GPUs, frameworks, load and memory" },
];

function Picture({ shot, className }: { shot: Shot; className: string }) {
  /* eslint-disable @next/next/no-img-element */
  if (!shot.light) return <img src={shot.dark} alt={shot.caption} className={className} loading="lazy" />;
  return (
    <>
      <img src={shot.light} alt={shot.caption} className={`${className} dark:hidden`} loading="lazy" />
      <img src={shot.dark} alt="" aria-hidden className={`${className} hidden dark:block`} loading="lazy" />
    </>
  );
  /* eslint-enable @next/next/no-img-element */
}

export function DashboardGallery() {
  const [open, setOpen] = useState<Shot | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(null);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = overflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <>
      <div className="mt-12 grid gap-6 md:grid-cols-3">
        {SHOTS.map((shot) => (
          <figure key={shot.dark}>
            <button
              type="button"
              onClick={() => setOpen(shot)}
              className="group relative block w-full cursor-zoom-in overflow-hidden rounded-xl border border-border shadow-frame focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              aria-label={`Enlarge: ${shot.caption}`}
            >
              <Picture shot={shot} className="aspect-[16/11] w-full object-cover object-left-top transition-transform duration-300 group-hover:scale-[1.03]" />
              <span className="absolute right-2 top-2 rounded-md border border-border bg-background/80 p-1.5 opacity-0 backdrop-blur transition-opacity group-hover:opacity-100">
                <ZoomIn className="h-3.5 w-3.5" />
              </span>
            </button>
            <figcaption className="mt-3 text-sm text-muted-foreground">{shot.caption}</figcaption>
          </figure>
        ))}
      </div>
      {open && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={open.caption}
          onClick={() => setOpen(null)}
          className="fixed inset-0 z-[100] flex cursor-zoom-out flex-col items-center justify-center gap-3 bg-black/85 p-4 backdrop-blur-sm sm:p-10"
        >
          <button
            type="button"
            onClick={() => setOpen(null)}
            className="absolute right-4 top-4 rounded-md border border-white/20 bg-black/60 p-2 text-white"
            aria-label="Close"
            autoFocus
          >
            <X className="h-4 w-4" />
          </button>
          <Picture shot={open} className="max-h-[85vh] w-auto max-w-full rounded-lg border border-white/10 object-contain" />
          <p className="text-sm text-white/70">{open.caption}</p>
        </div>
      )}
    </>
  );
}
