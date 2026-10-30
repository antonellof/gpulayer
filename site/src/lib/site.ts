// Canonical origin of the public site; metadata, the sitemap and robots.txt build URLs from it.
export const SITE_URL = "https://www.gpulayer.dev";

export const SITE_NAME = "GPUlayer";

export const SITE_DESCRIPTION =
  "Run any workload on every GPU and CPU you can reach: commands, Python functions and distributed PyTorch training on your laptop, remote servers, clusters and cloud VMs, with a web dashboard and an MCP server for AI agents. Open source, Apache-2.0.";

// The image link previews show (1200 x 630). public/og.png, made with next/og's ImageResponse.
export const OG_IMAGE = {
  url: "/og.png",
  width: 1200,
  height: 630,
  alt: "GPUlayer: run any workload on every GPU and CPU you can reach",
};
