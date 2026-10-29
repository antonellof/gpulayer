/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export for `npm run build` (writes a self-contained site to ./out). Dev mode runs
  // as a normal server, because Next 14's dev server rejects optional catch-all routes
  // under `output: "export"`.
  output: process.env.NODE_ENV === "production" ? "export" : undefined,
  // Separate build directory so `npm run build` never clobbers a running `npm run dev`.
  distDir: process.env.NODE_ENV === "production" ? ".next-build" : ".next",
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
  poweredByHeader: false,
};

export default nextConfig;
