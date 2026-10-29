import Link from "next/link";
import { SiteHeader } from "@/components/site/header";

export default function NotFound() {
  return (
    <div className="min-h-screen">
      <SiteHeader />
      <main className="mx-auto max-w-xl px-6 py-24 text-center">
        <h1 className="text-3xl font-semibold">Page not found</h1>
        <p className="mt-3 text-muted-foreground">This page does not exist or has moved.</p>
        <Link href="/docs/" className="mt-6 inline-block text-primary hover:underline">
          Go to the documentation
        </Link>
      </main>
    </div>
  );
}
