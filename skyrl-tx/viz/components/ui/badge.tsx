import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border border-transparent px-2 py-0.5 text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default: "bg-accent text-accent-foreground",
        secondary: "bg-muted text-muted-foreground uppercase tracking-wide",
        destructive: "bg-[var(--badge-gray-bg)] text-[var(--badge-gray-text)]",
        outline: "border border-border text-foreground",
        blue: "bg-[var(--badge-blue-bg)] text-[var(--badge-blue-text)] uppercase tracking-wide",
        green: "bg-[var(--badge-green-bg)] text-[var(--badge-green-text)] uppercase tracking-wide",
        gray: "bg-muted text-muted-foreground",
        purple: "bg-[var(--badge-purple-bg)] text-[var(--badge-purple-text)] uppercase tracking-wide",
        sampler: "bg-muted text-muted-foreground uppercase tracking-wide",
        training: "bg-muted text-muted-foreground uppercase tracking-wide",
        private: "bg-muted text-muted-foreground",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div
      data-slot="badge"
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  );
}

export { Badge, badgeVariants };
