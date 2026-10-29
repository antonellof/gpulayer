import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { JetBrains_Mono } from "next/font/google";
import { THEME_SCRIPT } from "@/components/site/theme-toggle";
import "./globals.css";

const code = JetBrains_Mono({ subsets: ["latin"], variable: "--font-code", display: "swap" });


export const metadata: Metadata = {
  title: {
    default: "GpuLayer | Run any GPU workload on any GPU you can reach",
    template: "%s | GpuLayer",
  },
  description:
    "A workload-agnostic GPU layer. Run commands in any language, Python functions with any library, and distributed jobs on your laptop, workstations, servers or cloud VMs, with a deeper integration for PyTorch. Pre-alpha.",
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
