import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center select-none whitespace-nowrap transition-colors disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-[2px] focus-visible:outline-offset-1 focus-visible:outline-[#0B99FF]",
  {
    variants: {
      variant: {
        default:
          "bg-[image:var(--btn-default-bg)] shadow-[var(--shadow-btn-inset)] hover:bg-[image:var(--btn-default-bg),linear-gradient(var(--overlay-hover),var(--overlay-hover))] hover:bg-blend-overlay active:bg-[image:var(--btn-default-bg),linear-gradient(var(--overlay-active),var(--overlay-active))] active:bg-blend-overlay text-foreground rounded-[var(--radius-control)]",
        primary:
          "bg-[image:var(--btn-primary-bg)] border border-[var(--btn-primary-border)] shadow-[var(--btn-primary-shadow)] text-white rounded-md hover:brightness-105 active:brightness-95 active:scale-[0.98] transition-all duration-200",
        secondary:
          "bg-[image:var(--btn-secondary-bg)] shadow-[var(--shadow-btn-inset)] text-white rounded-[var(--radius-control)]",
        danger:
          "bg-[image:var(--btn-danger-bg)] shadow-[var(--shadow-btn-inset)] text-white rounded-[var(--radius-control)]",
        outline: "border border-border bg-background hover:bg-muted rounded-md",
        ghost: "hover:bg-muted hover:text-foreground rounded-md",
        link: "text-accent underline-offset-4 hover:underline",
      },
      size: {
        default: "h-[var(--control-md)] px-[var(--px-control-md)] text-[var(--text-control-md)]",
        sm: "h-[var(--control-sm)] px-3 text-xs",
        xs: "h-[var(--control-xs)] px-2 text-xs",
        lg: "h-10 px-8 text-sm",
        icon: "h-[var(--control-md)] w-[var(--control-md)]",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
