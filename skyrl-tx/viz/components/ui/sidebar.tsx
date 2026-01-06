"use client";

import * as React from "react";
import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const Icons = {
  training: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  ),
  usage: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="20" x2="12" y2="10" />
      <line x1="18" y1="20" x2="18" y2="4" />
      <line x1="6" y1="20" x2="6" y2="16" />
    </svg>
  ),
  menu: () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  ),
  close: () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  ),
};

interface NavItemConfig {
  label: string;
  href: string;
  icon: keyof typeof Icons;
}

const navItems: NavItemConfig[] = [
  { label: "Overview", href: "/", icon: "usage" },
  { label: "Training Runs", href: "/training-runs", icon: "training" },
];

export function Sidebar() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  const getIsActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname === href;
  };

  const handleNavClick = () => {
    setIsOpen(false);
  };

  const sidebarContent = (
    <>
      <Link href="/" onClick={handleNavClick} className="flex items-center gap-2 px-4 py-4 border-b border-border hover:bg-sidebar-accent/50 transition-colors">
        <span className="font-semibold text-lg">SkyRL</span>
      </Link>

      <nav className="flex-1 py-2 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = Icons[item.icon];
          const isActive = getIsActive(item.href);

          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={handleNavClick}
              className={cn(
                "flex items-center gap-3 w-full px-4 py-2 text-sm transition-colors",
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                  : "text-sidebar-foreground hover:bg-sidebar-accent/50"
              )}
            >
              <Icon />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="px-4 py-4 text-center text-xs text-muted-foreground">
        <a href="https://github.com/SkyRL/skyrl-tx" target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
          SkyRL
        </a>
      </div>
    </>
  );

  return (
    <>
      <div className="md:hidden fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-4 py-3 bg-sidebar border-b border-border">
        <Link href="/" className="font-semibold">SkyRL</Link>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-2 text-sidebar-foreground hover:bg-sidebar-accent/50 rounded-md transition-colors"
          aria-label={isOpen ? "Close menu" : "Open menu"}
        >
          {isOpen ? <Icons.close /> : <Icons.menu />}
        </button>
      </div>

      {isOpen && (
        <div
          className="md:hidden fixed inset-0 z-40 bg-black/50"
          onClick={() => setIsOpen(false)}
        />
      )}

      <aside
        data-slot="sidebar-mobile"
        className={cn(
          "md:hidden fixed top-0 left-0 z-50 flex flex-col h-screen w-64 bg-sidebar border-r border-border transition-transform duration-200 ease-in-out",
          isOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {sidebarContent}
      </aside>

      <aside
        data-slot="sidebar"
        className="hidden md:flex flex-col h-screen w-[var(--sidebar-width)] border-r border-border bg-sidebar sticky top-0"
        style={{ "--sidebar-width": "16rem" } as React.CSSProperties}
      >
        {sidebarContent}
      </aside>
    </>
  );
}
