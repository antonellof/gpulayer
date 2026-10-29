import { PHASE_DEVELOPMENT_SERVER } from "next/constants.js";

/** @type {(phase: string) => import('next').NextConfig} */
export default function nextConfig(phase) {
  const isDev = phase === PHASE_DEVELOPMENT_SERVER;
  return {
    // `npm run build` writes a static export to ./out (build files in .next). The dev server
    // uses its own folder so a build never breaks a running `npm run dev`; it also runs as a
    // normal server because Next 14's dev server rejects optional catch-all routes under
    // `output: "export"`.
    output: isDev ? undefined : "export",
    distDir: isDev ? ".next-dev" : undefined,
    trailingSlash: true,
    images: { unoptimized: true },
    reactStrictMode: true,
    poweredByHeader: false,
  };
}
