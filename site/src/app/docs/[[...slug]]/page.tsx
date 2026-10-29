import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, ArrowRight, Pencil } from "lucide-react";
import { Markdown } from "@/components/docs/markdown";
import { Toc } from "@/components/docs/toc";
import { REPO, allDocs, extractHeadings, findDoc, navNeighbors, readDoc, routeHref } from "@/lib/docs";

interface PageProps {
  params: { slug?: string[] };
}

export const dynamicParams = false;

export function generateStaticParams() {
  return allDocs().map((d) => ({ slug: d.route }));
}

export function generateMetadata({ params }: PageProps): Metadata {
  const entry = findDoc(params.slug ?? []);
  return { title: entry ? entry.title : "Not found" };
}

export default function DocsPage({ params }: PageProps) {
  const entry = findDoc(params.slug ?? []);
  if (!entry) notFound();
  const content = readDoc(entry);
  const { prev, next } = navNeighbors(entry);
  const headings = content ? extractHeadings(content) : [];

  return (
    <div className="flex">
      <article className="min-w-0 flex-1 px-4 py-10 md:px-10 xl:px-14">
        {content ? (
          <Markdown content={content} file={entry.file} />
        ) : (
          <div className="docs-prose">
            <h1>{entry.title}</h1>
            <p>
              This page is being written. Follow progress in the{" "}
              <a href={`${REPO}/tree/main/docs`} target="_blank" rel="noreferrer">
                docs folder on GitHub
              </a>
              .
            </p>
          </div>
        )}

        <div className="mt-14 flex justify-end">
          <a
            href={`${REPO}/edit/main/docs/${entry.file}`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          >
            <Pencil className="h-3 w-3" /> Edit this page on GitHub
          </a>
        </div>

        {(prev || next) && (
          <div className="mt-6 grid gap-4 border-t border-border pt-6 sm:grid-cols-2">
            {prev ? (
              <Link href={routeHref(prev.route)} className="group rounded-lg border border-border p-4 hover:border-primary">
                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                  <ArrowLeft className="h-3 w-3" /> Previous
                </span>
                <span className="mt-1 block font-medium group-hover:text-primary">{prev.title}</span>
              </Link>
            ) : (
              <span />
            )}
            {next && (
              <Link
                href={routeHref(next.route)}
                className="group rounded-lg border border-border p-4 text-right hover:border-primary"
              >
                <span className="flex items-center justify-end gap-1 text-xs text-muted-foreground">
                  Next <ArrowRight className="h-3 w-3" />
                </span>
                <span className="mt-1 block font-medium group-hover:text-primary">{next.title}</span>
              </Link>
            )}
          </div>
        )}
      </article>
      <aside className="sticky top-14 hidden h-[calc(100vh-3.5rem)] w-60 shrink-0 overflow-y-auto py-10 pr-6 xl:block">
        <Toc items={headings} />
      </aside>
    </div>
  );
}
