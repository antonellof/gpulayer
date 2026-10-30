import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { JetBrains_Mono } from "next/font/google";
import { THEME_SCRIPT } from "@/components/site/theme-toggle";
import { OG_IMAGE, SITE_DESCRIPTION, SITE_NAME, SITE_URL } from "@/lib/site";
import "./globals.css";

const code = JetBrains_Mono({ subsets: ["latin"], variable: "--font-code", display: "swap" });


export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "GPUlayer | Run any workload on every GPU and CPU you can reach",
    template: "%s | GPUlayer",
  },
  description: SITE_DESCRIPTION,
  applicationName: SITE_NAME,
  keywords: [
    "GPU orchestration",
    "distributed training",
    "PyTorch",
    "PyTorch Lightning alternative",
    "DDP",
    "CUDA",
    "Apple Silicon",
    "MLX",
    "job scheduler",
    "MCP server",
    "open source",
  ],
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    siteName: SITE_NAME,
    url: "/",
    title: "GPUlayer | Run any workload on every GPU and CPU you can reach",
    description: SITE_DESCRIPTION,
    images: [OG_IMAGE],
  },
  twitter: {
    card: "summary_large_image",
    title: "GPUlayer | Run any workload on every GPU and CPU you can reach",
    description: SITE_DESCRIPTION,
    images: [OG_IMAGE.url],
  },
  robots: { index: true, follow: true },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${code.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="min-h-screen bg-background font-sans">{children}</body>
    </html>
  );
}
