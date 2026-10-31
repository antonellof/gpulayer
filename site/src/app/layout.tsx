import type { Metadata } from "next";
import Script from "next/script";
import { GeistSans } from "geist/font/sans";
import { JetBrains_Mono } from "next/font/google";
import { THEME_SCRIPT } from "@/components/site/theme-toggle";
import { GA_ID, OG_IMAGE, SITE_DESCRIPTION, SITE_NAME, SITE_URL } from "@/lib/site";
import "./globals.css";

const code = JetBrains_Mono({ subsets: ["latin"], variable: "--font-code", display: "swap" });


export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "GPUlayer | Train PyTorch on every GPU you can reach",
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
    title: "GPUlayer | Train PyTorch on every GPU you can reach",
    description: SITE_DESCRIPTION,
    images: [OG_IMAGE],
  },
  twitter: {
    card: "summary_large_image",
    title: "GPUlayer | Train PyTorch on every GPU you can reach",
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
      <body className="min-h-screen bg-background font-sans">
        {children}
        {/* Google Analytics (gtag.js), loaded after the page is interactive. */}
        <Script src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`} strategy="afterInteractive" />
        <Script id="gtag-init" strategy="afterInteractive">
          {`window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', '${GA_ID}');`}
        </Script>
      </body>
    </html>
  );
}
