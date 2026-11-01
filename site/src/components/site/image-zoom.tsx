"use client";

// Click to enlarge, site-wide: any element marked data-zoom (an <img>, or a box holding an
// inline SVG such as a mermaid diagram) opens full size over the page. Images inside links keep
// the link. Click, Escape or the close button closes it.

import { useEffect, useState } from "react";
import { X } from "lucide-react";

type Zoomed = { src?: string; html?: string; label: string };

export function ImageZoom() {
  const [item, setItem] = useState<Zoomed | null>(null);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      const el = (e.target as Element | null)?.closest?.("[data-zoom]");
      if (!el || el.closest("a")) return;
      e.preventDefault();
      // the dark twin of a light/dark pair has an empty alt: take the light one's
      const twin = el.previousElementSibling;
      if (el instanceof HTMLImageElement)
        setItem({ src: el.currentSrc || el.src, label: el.alt || (twin instanceof HTMLImageElement ? twin.alt : "") });
      else setItem({ html: el.innerHTML, label: el.getAttribute("aria-label") ?? "" });
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);

  useEffect(() => {
    if (!item) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setItem(null);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = overflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [item]);

  if (!item) return null;
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={item.label || "Enlarged image"}
      onClick={() => setItem(null)}
      className="fixed inset-0 z-[100] flex cursor-zoom-out flex-col items-center justify-center gap-3 bg-black/85 p-4 backdrop-blur-sm sm:p-10"
    >
      <button
        type="button"
        onClick={() => setItem(null)}
        className="absolute right-4 top-4 rounded-md border border-white/20 bg-black/60 p-2 text-white"
        aria-label="Close"
        autoFocus
      >
        <X className="h-4 w-4" />
      </button>
      <div className="max-h-[88vh] max-w-full overflow-auto rounded-lg border border-border bg-background p-2 sm:p-4">
        {item.src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={item.src} alt={item.label} className="max-h-[82vh] w-auto max-w-[92vw] object-contain" />
        ) : (
          <div
            className="[&_svg]:h-auto [&_svg]:max-h-[82vh] [&_svg]:w-[min(92vw,1400px)] [&_svg]:max-w-none"
            dangerouslySetInnerHTML={{ __html: item.html ?? "" }}
          />
        )}
      </div>
      {item.label && <p className="max-w-3xl text-center text-sm text-white/70">{item.label}</p>}
    </div>
  );
}
